import threading
import unittest

import torch

from h3lora.schedule import Schedule, ScheduleState, parse_rows, progress_from, resolve


class ScheduleTests(unittest.TestCase):
    def test_curves(self):
        for curve, midpoint in (
            ("linear", 0.5), ("cosine", 0.5), ("smoothstep", 0.5),
            ("power", 0.25), ("step", 1.0),
        ):
            with self.subTest(curve=curve):
                schedule = Schedule(start_strength=0.0, end_strength=1.0, curve=curve)
                self.assertEqual(schedule.evaluate(0.0), 0.0)
                self.assertAlmostEqual(schedule.evaluate(0.5), midpoint)
                self.assertEqual(schedule.evaluate(1.0), 1.0)

    def test_percent_clamping_and_explicit(self):
        schedule = Schedule(start_strength=1.0, end_strength=0.0, curve="linear",
                            start_percent=25.0, end_percent=75.0)
        self.assertEqual(schedule.evaluate(0.1), 1.0)
        self.assertAlmostEqual(schedule.evaluate(0.5), 0.5)
        self.assertEqual(schedule.evaluate(0.9), 0.0)
        explicit = Schedule(curve="explicit", explicit_strengths="1, .25  -0.5")
        self.assertEqual([explicit.evaluate(p) for p in (0.0, 0.5, 1.0)],
                         [1.0, 0.25, -0.5])

    def test_explicit_rejects_invalid_values(self):
        for value in ("", "1,nope,0", "1,nan,0", "1,inf,0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Schedule(curve="explicit", explicit_strengths=value)

    def test_rows_and_later_chain_link_wins(self):
        all_rows = parse_rows("all")
        singles = parse_rows("1,3")
        interval = parse_rows("2-4")
        self.assertTrue(all_rows(99))
        self.assertTrue(singles(1) and singles(3) and not singles(2))
        self.assertTrue(all(interval(row) for row in (2, 3, 4)) and not interval(1))

        first = Schedule(rows="all", start_strength=1.0)
        second = Schedule(rows="2-4", start_strength=2.0)
        self.assertIs(resolve((first, second), 1), first)
        self.assertIs(resolve((first, second), 3), second)

    def test_progress_exact_match_sigma_domain_and_fallback(self):
        options = {"sample_sigmas": torch.tensor([10.0, 7.0, 2.0, 0.0])}
        self.assertEqual(progress_from(torch.tensor([10.0]), options), 0.0)
        self.assertAlmostEqual(progress_from(torch.tensor([7.0]), options), 0.5)
        self.assertEqual(progress_from(torch.tensor([2.0]), options), 1.0)
        self.assertAlmostEqual(progress_from(torch.tensor([7.0]), options, "sigma"), 0.3)
        self.assertAlmostEqual(progress_from(torch.tensor([5.0]), options), 0.5)

    def test_schedule_state_is_thread_local(self):
        state = ScheduleState()
        barrier = threading.Barrier(2)
        seen = []

        def worker(value):
            state.set({"layer": torch.tensor([value])}, {})
            barrier.wait()
            seen.append(float(state.scales_for("layer")[0]))

        threads = [threading.Thread(target=worker, args=(value,)) for value in (1.0, 2.0)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(sorted(seen), [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
