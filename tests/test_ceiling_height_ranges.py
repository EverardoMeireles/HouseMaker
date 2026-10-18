# ### Imports ###
from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from housemaker.ceiling_height_ranges import (
    CeilingHeightVideoRange,
    sample_marked_video_frames,
)


# ### Marked-range validation tests ###
class CeilingHeightVideoRangeTests(unittest.TestCase):
    def test_range_has_inclusive_integer_bounds_and_is_immutable(self) -> None:
        marked = CeilingHeightVideoRange(2, 7, 7)

        self.assertEqual(marked.start_frame, marked.end_frame)
        with self.assertRaises(FrozenInstanceError):
            marked.start_frame = 8  # type: ignore[misc]

    def test_ranges_reject_non_integer_values_and_reversed_bounds(self) -> None:
        for arguments in ((True, 0, 1), (0, 0.5, 1), (0, 0, False)):
            with self.subTest(arguments=arguments), self.assertRaises(TypeError):
                CeilingHeightVideoRange(*arguments)
        for arguments in ((0, -1, 3), (0, 3, 2)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                CeilingHeightVideoRange(*arguments)

    def test_level_identifiers_need_not_be_positive(self) -> None:
        self.assertEqual(CeilingHeightVideoRange(-1, 0, 10).level_index, -1)


# ### Marked-range sampling tests ###
class MarkedVideoSamplingTests(unittest.TestCase):
    def test_samples_only_marks_in_chronological_order_with_endpoints(self) -> None:
        samples = sample_marked_video_frames(
            (
                CeilingHeightVideoRange(2, 30, 40),
                CeilingHeightVideoRange(1, 4, 15),
            ),
            interval=5,
            frame_count=60,
        )

        self.assertEqual(
            samples,
            ((4, 1), (9, 1), (14, 1), (15, 1), (30, 2), (35, 2), (40, 2)),
        )

    def test_one_level_can_have_multiple_disjoint_marks(self) -> None:
        samples = sample_marked_video_frames(
            (CeilingHeightVideoRange(1, 0, 1), CeilingHeightVideoRange(1, 10, 12)),
            interval=15,
            frame_count=20,
        )

        self.assertEqual(samples, ((0, 1), (1, 1), (10, 1), (12, 1)))

    def test_same_level_overlaps_deduplicate_and_retain_each_mark_end(self) -> None:
        samples = sample_marked_video_frames(
            (
                CeilingHeightVideoRange(1, 0, 10),
                CeilingHeightVideoRange(1, 5, 15),
                CeilingHeightVideoRange(1, 0, 10),
            ),
            interval=5,
            frame_count=20,
        )

        self.assertEqual(samples, ((0, 1), (5, 1), (10, 1), (15, 1)))

    def test_same_level_overlaps_keep_different_sampling_offsets(self) -> None:
        samples = sample_marked_video_frames(
            (CeilingHeightVideoRange(1, 0, 10), CeilingHeightVideoRange(1, 3, 7)),
            interval=5,
            frame_count=20,
        )

        self.assertEqual(samples, ((0, 1), (3, 1), (5, 1), (7, 1), (10, 1)))

    def test_different_levels_cannot_share_any_frame_even_between_samples(self) -> None:
        for second in (
            CeilingHeightVideoRange(2, 10, 20),
            CeilingHeightVideoRange(2, 4, 6),
        ):
            with self.subTest(second=second), self.assertRaisesRegex(
                ValueError, "different levels cannot overlap"
            ):
                sample_marked_video_frames(
                    (CeilingHeightVideoRange(1, 0, 10), second),
                    interval=100,
                    frame_count=30,
                )

    def test_adjacent_non_overlapping_level_marks_are_valid(self) -> None:
        self.assertEqual(
            sample_marked_video_frames(
                (CeilingHeightVideoRange(1, 0, 10), CeilingHeightVideoRange(2, 11, 20)),
                interval=100,
                frame_count=21,
            ),
            ((0, 1), (10, 1), (11, 2), (20, 2)),
        )

    def test_mark_beyond_video_is_rejected_not_silently_cropped(self) -> None:
        with self.assertRaisesRegex(ValueError, "beyond the loaded video"):
            sample_marked_video_frames((CeilingHeightVideoRange(1, 0, 20),), 5, 20)

    def test_sampling_rejects_invalid_intervals_counts_and_range_types(self) -> None:
        for interval, count in ((0, 20), (-1, 20), (5, -1)):
            with self.subTest(interval=interval, count=count), self.assertRaises(ValueError):
                sample_marked_video_frames((), interval, count)
        for interval, count in ((True, 20), (5, 20.0), (5, False)):
            with self.subTest(interval=interval, count=count), self.assertRaises(TypeError):
                sample_marked_video_frames((), interval, count)
        with self.assertRaises(TypeError):
            sample_marked_video_frames(((1, 0, 10),), 5, 20)

    def test_no_marks_yields_no_billable_samples(self) -> None:
        self.assertEqual(sample_marked_video_frames((), 15, 100), ())
        self.assertEqual(sample_marked_video_frames((), 15, 0), ())


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
