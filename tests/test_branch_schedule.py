from collections import OrderedDict
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from comfy import ops
from comfy.ldm.minimax.model import MLP
from comfy.model_patcher import ModelPatcher
from comfy.quant_ops import QuantizedTensor

import h3lora.branch as branch_mod
from h3lora.branch import FusedBranch, LoraBank, LoraBranch, ScheduleController, fuse
from h3lora.schedule import Schedule, ScheduleState


class _BankPatcher:
    def __init__(self, model):
        self.model = model


class BranchScheduleTests(unittest.TestCase):
    def test_unscheduled_fusion_is_original_formula(self):
        up1, down1 = torch.randn(4, 2), torch.randn(2, 3)
        up2, down2 = torch.randn(4, 1), torch.randn(1, 3)
        got_up, got_down = fuse(
            [(up1, down1, 0.75), (up2, down2, -0.25)], torch.float32)
        expected_up = torch.cat([up1.float() * 0.75, up2.float() * -0.25], dim=1).contiguous()
        expected_down = torch.cat([down1.float(), down2.float()], dim=0).contiguous()
        self.assertTrue(torch.equal(got_up, expected_up))
        self.assertTrue(torch.equal(got_down, expected_down))

    def test_scheduled_rank_scale_and_bias(self):
        schedule = Schedule(start_strength=0.0, end_strength=1.0)
        fused = fuse([(
            torch.tensor([[2.0, 0.0]]),
            torch.tensor([[3.0], [0.0]]),
            1.0,
            schedule,
            torch.tensor([2.0]),
        )], torch.float32)
        bank = LoraBank(OrderedDict([("layer", fused)]))
        state = ScheduleState()
        state.set({"layer": torch.tensor([0.5, 0.5])},
                  {"layer": torch.tensor([0.5])})
        branch = LoraBranch(_BankPatcher(bank), state, "layer", lambda value: value)
        self.assertTrue(torch.equal(branch(torch.tensor([[1.0]])), torch.tensor([[5.0]])))

    def test_two_controllers_nest_without_clobbering(self):
        schedule_a = Schedule(start_strength=1.0, end_strength=0.0)
        schedule_b = Schedule(start_strength=0.0, end_strength=2.0)
        fused_a = FusedBranch(torch.empty(0), torch.empty(0), [(0, 1, schedule_a, 1.0)], [])
        fused_b = FusedBranch(torch.empty(0), torch.empty(0), [(0, 1, schedule_b, 1.0)], [])
        state_a, state_b = ScheduleState(), ScheduleState()
        controller_a = ScheduleController(state_a, {"a": fused_a}, torch.float32)
        controller_b = ScheduleController(state_b, {"b": fused_b}, torch.float32)
        options = {"sample_sigmas": torch.tensor([10.0, 5.0, 0.0])}

        def original(*_args, **_kwargs):
            return (float(state_a.scales_for("a")[0]),
                    float(state_b.scales_for("b")[0]))

        def inner(*args, **kwargs):
            return controller_b(original, *args, **kwargs)

        result = controller_a(inner, None, torch.tensor([5.0]),
                              transformer_options=options)
        self.assertEqual(result, (0.0, 2.0))
        self.assertIsNone(state_a.scales_for("a"))
        self.assertIsNone(state_b.scales_for("b"))

    def test_chunked_bias_broadcasts_when_seq_equals_out(self):
        # bias_term is [out], not [seq]. When those lengths match, slicing the
        # bias on dim 0 looks like a row slice and corrupts every token.
        seq = out = 6
        rank, hidden = 2, 3
        up = torch.zeros(out, rank)
        up[:, 0] = 1.0
        down = torch.zeros(rank, hidden)
        down[0, 0] = 1.0
        bias = torch.arange(1, out + 1, dtype=torch.float32)
        fused = fuse([(up, down, 1.0, None, bias)], torch.float32)
        bank = LoraBank(OrderedDict([("layer", fused)]))
        state = ScheduleState()
        state.set({"layer": torch.ones(rank)}, {"layer": torch.ones(1)})

        def original(value):
            return torch.zeros(value.shape[0], out)

        branch = LoraBranch(_BankPatcher(bank), state, "layer", original)
        old_min, old_slice = branch_mod._WHOLE_RESIDUAL_MAX_BYTES, branch_mod._RESIDUAL_SLICE_BYTES
        branch_mod._WHOLE_RESIDUAL_MAX_BYTES = 0
        branch_mod._RESIDUAL_SLICE_BYTES = 1
        try:
            got = branch(torch.ones(seq, hidden))
        finally:
            branch_mod._WHOLE_RESIDUAL_MAX_BYTES = old_min
            branch_mod._RESIDUAL_SLICE_BYTES = old_slice
        expected = (1.0 + bias).expand(seq, out)
        self.assertTrue(torch.allclose(got, expected))


    def test_grouped_fc2_branch_uses_fused_base_and_keeps_stacked_adapters(self):
        # More than one slice, plus a tail. Start with a real H3 MLP whose
        # quantized fc2 goes through linear_input_act.
        device = torch.device("cpu")
        dtype = torch.float32
        dtype_ops = ops.mixed_precision_ops(compute_dtype=dtype)
        mlp = MLP(64, 256, dtype=dtype, device=device, operations=dtype_ops)
        mlp.fc1 = torch.nn.Linear(64, 512, bias=False)
        layer = torch.nn.Module()
        layer.block = torch.nn.Module()
        layer.block.mlp = mlp
        packed = QuantizedTensor.from_float(
            torch.randn(64, 256), "AsymW4A8Int8Layout", bits=4, scale_search=False)
        mlp.fc2.weight = torch.nn.Parameter(packed, requires_grad=False)
        mlp.fc2.layout_type = "AsymW4A8Int8Layout"
        mlp.fc2.quant_format = "asym_w4a8_int8"
        codes = packed.state_dict()[""].clone()
        x = torch.randn(15, 64)
        a, b = torch.randn(2, 256), torch.randn(64, 2)
        c, d = torch.randn(3, 256), torch.randn(64, 3)
        with torch.inference_mode():
            baseline = mlp(x)
            gate, up = mlp.fc1(x).chunk(2, dim=-1)
            activated = F.silu(gate).mul_(up)
            expected = baseline + F.linear(F.linear(activated, a), b)
            expected += F.linear(F.linear(activated, c), d) * 0.5
            first = ModelPatcher(layer, device, device)
            with patch.object(branch_mod.comfy.model_management, "get_torch_device",
                              return_value=device):
                branch_mod.attach(first, {"block.mlp.fc2": [(b, a, 1.0)]}, dtype, "first")
                second = first.clone()
                _, controller = branch_mod.attach(second, {"block.mlp.fc2": [
                    (d, c, 1.0, Schedule(start_strength=0.5, end_strength=0.5), None),
                ]}, dtype, "second")
            self.assertNotIn("block.mlp.fc2.forward", second.object_patches)
            self.assertNotIn("block.mlp.fc2.comfy_force_forward", second.object_patches)
            self.assertEqual(len(first.object_patches["block.mlp.forward"].branches), 1)
            self.assertEqual(len(second.object_patches["block.mlp.forward"].branches), 2)
            second.patch_model(load_weights=False)
            try:
                with patch.object(branch_mod, "_FC2_SLICE_ROWS", 7), patch.object(
                        ops, "linear_input_act", wraps=ops.linear_input_act) as base_call:
                    result = controller(lambda *_args, **_kwargs: mlp(x), None,
                                        torch.tensor([0.5]))
                    base_call.assert_called_once()
                    self.assertEqual(base_call.call_args.args[2], "swiglu")
                torch.testing.assert_close(result, expected, atol=2e-5, rtol=2e-5)
                self.assertTrue(torch.equal(mlp.fc2.weight.state_dict()[""], codes))
            finally:
                second.unpatch_model(unpatch_weights=False)
            torch.testing.assert_close(mlp(x), baseline, atol=0, rtol=0)

            # A custom forward can still call the fused path and skip fc2.forward.
            native_forward = mlp.forward
            with patch.object(mlp, "forward", lambda value: native_forward(value)):
                fallback = ModelPatcher(layer, device, device)
                notes = []
                with patch.object(branch_mod.comfy.model_management, "get_torch_device",
                                  return_value=device):
                    with self.assertLogs("h3.powerlorastack", level="WARNING") as logs:
                        branch_mod.attach(fallback, {"block.mlp.fc2": [(b, a, 1.0)]},
                                          dtype, "fallback", notes=notes)
                self.assertIn("block.mlp.fc2", logs.output[0])
                self.assertIn("may bypass its LoRA branch", logs.output[0])
                self.assertEqual(len(notes), 1)
                self.assertIn("block.mlp.fc2", notes[0])
                self.assertIn("block.mlp.fc2.forward", fallback.object_patches)
                fallback.patch_model(load_weights=False)
                try:
                    torch.testing.assert_close(mlp(x), baseline, atol=0, rtol=0)
                finally:
                    fallback.unpatch_model(unpatch_weights=False)

    def test_grouped_fc2_branch_matches_reference_for_all_widths(self):
        device = torch.device("cpu")
        dtype = torch.float32
        dtype_ops = ops.mixed_precision_ops(compute_dtype=dtype)
        for bits in (3, 4, 5, 6):
            mlp = MLP(64, 256, dtype=dtype, device=device, operations=dtype_ops)
            mlp.fc1 = torch.nn.Linear(64, 512, bias=False)
            layer = torch.nn.Module()
            layer.block = torch.nn.Module()
            layer.block.mlp = mlp
            packed = QuantizedTensor.from_float(
                torch.randn(64, 256), "AsymW4A8Int8Layout", bits=bits,
                scale_search=False)
            mlp.fc2.weight = torch.nn.Parameter(packed, requires_grad=False)
            mlp.fc2.layout_type = "AsymW4A8Int8Layout"
            mlp.fc2.quant_format = f"w{bits}a8_int8"
            a, b = torch.randn(2, 256), torch.randn(64, 2)
            x = torch.randn(15, 64)
            with torch.inference_mode():
                baseline = mlp(x)
                gate, up = mlp.fc1(x).chunk(2, dim=-1)
                activated = F.silu(gate).mul_(up)
                expected = baseline + F.linear(F.linear(activated, a), b)
            patcher = ModelPatcher(layer, device, device)
            with patch.object(branch_mod.comfy.model_management, "get_torch_device",
                              return_value=device):
                branch_mod.attach(patcher, {"block.mlp.fc2": [(b, a, 1.0)]},
                                  dtype, f"w{bits}")
            patcher.patch_model(load_weights=False)
            try:
                with patch.object(branch_mod, "_FC2_SLICE_ROWS", 7):
                    result = mlp(x)
                torch.testing.assert_close(result, expected, atol=2e-5, rtol=2e-5)
            finally:
                patcher.unpatch_model(unpatch_weights=False)


if __name__ == "__main__":
    unittest.main()
