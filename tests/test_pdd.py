import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from h3lora.pdd import PddBank, PddFinalLayer, PddState, _as_bank, _blend, interval, peel


class PeelTests(unittest.TestCase):
    def test_stacked_set_weight(self):
        sd = {
            "diffusion_model.final_layer.video_out.set_weight": torch.randn(3072, 5376),
            "diffusion_model.final_layer.video_out.set_bias": torch.randn(3072),
            "diffusion_model.final_layer.audio_out.set_weight": torch.randn(1024, 5376),
            "diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight": torch.randn(8, 5376),
        }
        bank, note = peel(sd, 96, 32, 5376)
        self.assertIsNotNone(bank)
        self.assertEqual(bank["n"], 32)
        self.assertEqual(bank["video_w"].shape, (3072, 5376))
        self.assertEqual(bank["audio_w"].shape, (1024, 5376))
        self.assertIn("n=32", note)
        self.assertNotIn("set_weight", sd)
        self.assertIn("diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight", sd)

    def test_proj_out_3d(self):
        sd = {
            "proj_out.weight": torch.randn(32, 96, 5376),
            "proj_out.bias": torch.randn(32, 96),
            "audio_proj_out.weight": torch.randn(32, 32, 5376),
            "audio_proj_out.bias": torch.randn(32, 32),
        }
        bank, note = peel(sd, 96, 32, 5376)
        self.assertEqual(bank["n"], 32)
        self.assertEqual(bank["video_w"].shape, (3072, 5376))
        self.assertEqual(bank["video_b"].shape, (3072,))
        self.assertEqual(bank["audio_w"].shape, (1024, 5376))
        self.assertFalse(sd)

    def test_native_head_left_alone(self):
        w = torch.randn(96, 5376)
        sd = {"diffusion_model.final_layer.video_out.set_weight": w}
        bank, note = peel(sd, 96, 32, 5376)
        self.assertIsNone(bank)
        self.assertEqual(note, "")
        self.assertIs(sd["diffusion_model.final_layer.video_out.set_weight"], w)

    def test_garbage_dropped(self):
        sd = {"diffusion_model.final_layer.video_out.set_weight": torch.randn(100, 5376)}
        bank, note = peel(sd, 96, 32, 5376)
        self.assertIsNone(bank)
        self.assertIn("dropped", note)
        self.assertFalse(sd)


class BlendTests(unittest.TestCase):
    def test_as_bank_3d_and_2d(self):
        t = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
        n, flat = _as_bank(t, 4)
        self.assertEqual(n, 2)
        self.assertEqual(flat.shape, (8, 3))
        n2, flat2 = _as_bank(flat, 4)
        self.assertEqual(n2, 2)
        self.assertTrue(torch.equal(flat, flat2))

    def test_single_interval_is_that_head(self):
        n, o, hidden = 4, 2, 3
        weight = torch.zeros(n * o, hidden)
        for i in range(n):
            weight[i * o:(i + 1) * o] = float(i + 1)
        h = torch.ones(5, hidden)
        out = _blend(weight, None, h, n, 2, 3, 1.0)
        self.assertEqual(out.shape, (5, o))
        self.assertTrue(torch.allclose(out, torch.full((5, o), 3.0 * hidden)))

    def test_interval_monotonic(self):
        start, stop = interval(torch.tensor(1.0), torch.tensor(0.0), 32, 12.0)
        self.assertGreaterEqual(stop, start + 1)
        self.assertGreaterEqual(start, 0)
        self.assertLessEqual(stop, 32)

    def test_partial_audio_bank_falls_back_to_native_video(self):
        sd = {
            "final_layer.audio_out.set_weight": torch.randn(2 * 2, 3),
        }
        bank, note = peel(sd, 4, 2, 3)
        self.assertIsNotNone(bank)
        self.assertIsNone(bank["video_w"])
        self.assertIn("n=2", note)

    def test_strength_interpolates_native_and_pdd_heads(self):
        class StaticAdaln(nn.Module):
            def forward(self, _t):
                return torch.zeros(2, 3), torch.zeros(2, 3)

        inner = SimpleNamespace(
            norm=nn.Identity(),
            adaln_proj=StaticAdaln(),
            video_out=nn.Linear(3, 2),
            audio_out=nn.Linear(3, 1),
        )
        bank, _ = peel({
            "final_layer.video_out.set_weight": torch.ones(2 * 2, 3),
        }, 2, 1, 3)
        state = PddState(0.5, None)
        state.set(torch.tensor(1.0), torch.tensor([1.0, 0.0]), (12.0, 3.0), 0.5)
        layer = PddFinalLayer(inner, PddBank(bank), state)
        video, audio = layer(
            torch.ones(4, 3), torch.zeros(1, 3), (0, 2, 0), (2, 4, 1))
        native = inner.video_out(torch.ones(2, 3))
        pdd = torch.full_like(native, 3.0)
        self.assertTrue(torch.allclose(video, native + 0.5 * (pdd - native)))
        self.assertEqual(audio.shape, (2, 1))

    def test_current_comfy_signature_prefers_direct_schedule_arguments(self):
        class StaticAdaln(nn.Module):
            def forward(self, _t):
                return torch.zeros(2, 3), torch.zeros(2, 3)

        inner = SimpleNamespace(
            norm=nn.Identity(),
            adaln_proj=StaticAdaln(),
            video_out=nn.Linear(3, 2),
            audio_out=nn.Linear(3, 1),
        )
        video_bank = torch.cat([
            torch.full((2, 3), float(head))
            for head in (1, 2, 3, 4)
        ])
        bank, _ = peel({
            "final_layer.video_out.set_weight": video_bank,
        }, 2, 1, 3)

        state = PddState(1.0, None)
        # Deliberately unusable legacy schedule data: direct arguments from
        # current ComfyUI must take precedence, while strength still comes from
        # the outer stack wrapper's state.
        state.set(torch.tensor(0.5), torch.empty(0), (1.0, 1.0), 1.0)
        layer = PddFinalLayer(inner, PddBank(bank), state)

        sigma = torch.tensor(1.0)
        sample_sigmas = torch.tensor([1.0, 0.0])
        shifts = (12.0, 3.0)
        x = torch.ones(4, 3)
        video, audio = layer(
            x,
            torch.zeros(1, 3),
            (0, 2, 0),
            (2, 4, 1),
            sigma,
            sample_sigmas,
            shifts,
        )

        start, stop = interval(sigma, sample_sigmas[1], 4, shifts[0])
        expected_video = _blend(
            layer.bank.video_w,
            layer.bank.video_b,
            x[:2],
            4,
            start,
            stop,
            shifts[0],
        )
        self.assertTrue(torch.allclose(video, expected_video))
        self.assertEqual(audio.shape, (2, 1))


if __name__ == "__main__":
    unittest.main()
