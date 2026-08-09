from collections import OrderedDict
import unittest

import torch

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


if __name__ == "__main__":
    unittest.main()
