# ### Imports ###
from __future__ import annotations

import unittest

from housemaker.camera_models import CameraPose
from housemaker.generation_state import MASK_MODE_PAINT, MaskPoint, MaskStroke
from housemaker.surface_texture_state import (
    SURFACE_TYPE_CEILING,
    SURFACE_TYPE_FLOOR,
    SURFACE_TYPE_WALL,
    SurfaceTextureAssignment,
    SurfaceTextureData,
)
from housemaker.video_source import VideoMetadata


# ### Fixture helpers ###
def _stroke(x: float = 0.5) -> MaskStroke:
    return MaskStroke(
        mode=MASK_MODE_PAINT,
        radius_normalized=0.1,
        points=(MaskPoint(x=x, y=0.5),),
    )


def _video(frame_count: int = 12) -> VideoMetadata:
    return VideoMetadata(
        path="house.mp4",
        frame_count=frame_count,
        fps=24.0,
        width=1280,
        height=720,
    )


def _assignment() -> SurfaceTextureAssignment:
    return SurfaceTextureAssignment(
        assignment_id="texture-1",
        surface_type=SURFACE_TYPE_WALL,
        surface_ids=(
            "level:2/room:0/wall:1:2",
            "level:2/room:0/wall:2:3",
        ),
        provider="openai",
        provider_task_id="response-123",
        asset_path="surface_textures/texture-1.png",
        combined_area_m2=12.75,
        area_description="Two connected interior walls",
        reference_frame_indices=(2, 7),
        texture_width=1024,
        texture_height=1024,
    )


# ### State round-trip tests ###
class SurfaceTextureStateRoundTripTests(unittest.TestCase):
    def test_3d_texture_mask_strokes_round_trip_per_stable_surface(self) -> None:
        surface_id = "level:2/room:5/wall:1:2"
        state = SurfaceTextureData(
            texture_mask_strokes={surface_id: [_stroke(0.25)]}
        )

        restored = SurfaceTextureData.from_dict(state.to_dict())

        self.assertEqual(restored.strokes_for_surface(surface_id), [_stroke(0.25)])
        restored.set_surface_strokes(surface_id, [])
        self.assertEqual(restored.texture_mask_strokes, {})

    def test_video_enclosed_fill_action_round_trips(self) -> None:
        fill = MaskStroke(
            mode=MASK_MODE_PAINT,
            radius_normalized=0.000001,
            points=(MaskPoint(x=0.5, y=0.5),),
            is_fill=True,
        )
        state = SurfaceTextureData(
            video_metadata=_video(),
            frame_strokes={2: [fill]},
        )

        restored = SurfaceTextureData.from_dict(state.to_dict())

        self.assertEqual(restored.strokes_for_frame(2), [fill])

    def test_complete_state_round_trips_and_clones_independently(self) -> None:
        original = SurfaceTextureData(
            video_metadata=_video(),
            current_frame_index=7,
            frame_strokes={2: [_stroke(0.25)], 7: [_stroke(0.75)]},
            camera_pose=CameraPose(
                x=1.0,
                y=2.0,
                z=1.65,
                yaw_degrees=35.0,
                pitch_degrees=-4.0,
                fov_degrees=68.0,
            ),
            selected_surface_type=SURFACE_TYPE_WALL,
            selected_surface_ids=(
                "level:2/room:0/wall:1:2",
                "level:2/room:0/wall:2:3",
            ),
            assignments=[_assignment()],
        )

        loaded = SurfaceTextureData.from_dict(original.to_dict())
        cloned = original.clone()
        cloned.set_frame_strokes(2, [])
        cloned.assignments.clear()

        self.assertEqual(loaded, original)
        self.assertEqual(original.strokes_for_frame(2), [_stroke(0.25)])
        self.assertEqual(original.assignments, [_assignment()])
        self.assertEqual(original.generated_assignments, original.assignments)
        self.assertEqual(
            original.assignments_for_surface("level:2/room:0/wall:2:3"),
            [_assignment()],
        )

    def test_multi_frame_strokes_are_isolated_and_empty_frames_are_removed(self) -> None:
        state = SurfaceTextureData(video_metadata=_video(3))

        state.set_frame_strokes(0, [_stroke(0.2)])
        state.set_frame_strokes(1, [_stroke(0.8)])
        state.set_frame_strokes(0, [])

        self.assertEqual(state.strokes_for_frame(0), [])
        self.assertEqual(state.strokes_for_frame(1), [_stroke(0.8)])
        returned = state.strokes_for_frame(1)
        returned.clear()
        self.assertEqual(state.strokes_for_frame(1), [_stroke(0.8)])


# ### Selection tests ###
class SurfaceTextureSelectionTests(unittest.TestCase):
    def test_selection_accepts_exact_geometry_ids_and_deduplicates_in_order(self) -> None:
        state = SurfaceTextureData()

        state.set_selection(
            SURFACE_TYPE_FLOOR,
            [
                "level:2/room:0/floor",
                "level:2/room:1/floor",
                "level:2/room:0/floor",
            ],
        )

        self.assertEqual(state.selected_surface_type, SURFACE_TYPE_FLOOR)
        self.assertEqual(
            state.selected_surface_ids,
            ("level:2/room:0/floor", "level:2/room:1/floor"),
        )
        state.set_selection(SURFACE_TYPE_CEILING, [])
        self.assertIsNone(state.selected_surface_type)
        self.assertEqual(state.selected_surface_ids, ())

    def test_selection_accepts_level_fallback_surface_ids_without_rooms(self) -> None:
        for surface_type, surface_id in (
            (SURFACE_TYPE_WALL, "level:2/wall:1:2"),
            (SURFACE_TYPE_FLOOR, "level:2/floor"),
            (SURFACE_TYPE_CEILING, "level:2/ceiling"),
        ):
            with self.subTest(surface_type=surface_type):
                state = SurfaceTextureData(
                    selected_surface_type=surface_type,
                    selected_surface_ids=(surface_id,),
                )
                self.assertEqual(state.selected_surface_ids, (surface_id,))

    def test_selection_rejects_mixed_types_and_malformed_ids(self) -> None:
        invalid_selections = (
            (
                SURFACE_TYPE_WALL,
                ("level:2/room:0/wall:1:2", "level:2/room:0/floor"),
            ),
            (SURFACE_TYPE_CEILING, ("level:2/room:0/floor",)),
            (SURFACE_TYPE_WALL, ("room:0/wall:1:2",)),
            (SURFACE_TYPE_WALL, ("level:-1/room:0/wall:1:2",)),
        )
        for surface_type, surface_ids in invalid_selections:
            with self.subTest(surface_type=surface_type, surface_ids=surface_ids):
                with self.assertRaises(ValueError):
                    SurfaceTextureData(
                        selected_surface_type=surface_type,
                        selected_surface_ids=surface_ids,
                    )


# ### Assignment validation tests ###
class SurfaceTextureAssignmentTests(unittest.TestCase):
    def test_assignment_normalizes_safe_relative_paths_and_legacy_aliases(self) -> None:
        assignment = SurfaceTextureAssignment.from_dict(
            {
                "id": "legacy-texture",
                "surfaces": ["level:2/room:0/ceiling"],
                "provider": "test-provider",
                "task_id": "task-1",
                "path": "textures\\ceiling.png",
                "area_m2": 8.5,
                "frame_indices": [4, 4, 6],
                "width": 512,
                "height": 256,
            }
        )

        self.assertEqual(assignment.surface_type, SURFACE_TYPE_CEILING)
        self.assertEqual(assignment.asset_path, "textures/ceiling.png")
        self.assertEqual(assignment.reference_frame_indices, (4, 6))
        self.assertEqual((assignment.texture_width, assignment.texture_height), (512, 256))

    def test_assignment_rejects_unsafe_assets_invalid_area_and_partial_dimensions(self) -> None:
        base_arguments = {
            "assignment_id": "texture-1",
            "surface_type": SURFACE_TYPE_FLOOR,
            "surface_ids": ("level:2/room:0/floor",),
            "provider": "provider",
            "asset_path": "textures/floor.png",
        }
        invalid_overrides = (
            {"asset_path": "../floor.png"},
            {"asset_path": "C:\\textures\\floor.png"},
            {"asset_path": "/textures/floor.png"},
            {"combined_area_m2": -0.1},
            {"combined_area_m2": float("nan")},
            {"reference_frame_indices": (-1,)},
            {"texture_width": 512, "texture_height": None},
            {"texture_width": 0, "texture_height": 512},
        )
        for overrides in invalid_overrides:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    SurfaceTextureAssignment(**(base_arguments | overrides))


# ### Defensive loading tests ###
class SurfaceTextureDefensiveLoadingTests(unittest.TestCase):
    def test_legacy_keys_load_and_out_of_range_frame_data_is_bounded(self) -> None:
        payload = {
            "video": _video(3).to_dict(),
            "current_frame_index": 99,
            "first_person_camera_pose": CameraPose(x=3.0).to_dict(),
            "frame_strokes": {
                "1": [_stroke().to_dict()],
                "99": [_stroke().to_dict()],
                "bad": [_stroke().to_dict()],
            },
            "selected_surface_ids": ["level:2/room:0/floor"],
            "generated_assignments": [_assignment().to_dict()],
        }

        loaded = SurfaceTextureData.from_dict(payload)

        self.assertEqual(loaded.current_frame_index, 2)
        self.assertEqual(loaded.strokes_for_frame(1), [_stroke()])
        self.assertEqual(set(loaded.frame_strokes), {1})
        self.assertEqual(loaded.camera_pose, CameraPose(x=3.0))
        self.assertEqual(loaded.selected_surface_type, SURFACE_TYPE_FLOOR)
        self.assertEqual(loaded.assignments, [_assignment()])

    def test_malformed_optional_records_fall_back_without_losing_valid_records(self) -> None:
        valid_assignment = _assignment().to_dict()
        duplicate_assignment = _assignment().to_dict()
        malformed_assignment = _assignment().to_dict() | {
            "assignment_id": "unsafe",
            "asset_path": "../../outside.png",
        }
        loaded = SurfaceTextureData.from_dict(
            {
                "video_metadata": {"frame_count": "bad"},
                "current_frame_index": -4,
                "camera_pose": {"x": float("nan")},
                "frame_strokes": {
                    "0": [
                        _stroke().to_dict(),
                        {"mode": "unknown"},
                    ]
                },
                "selected_surface_type": SURFACE_TYPE_WALL,
                "selected_surface_ids": ["not-a-surface"],
                "assignments": [
                    valid_assignment,
                    malformed_assignment,
                    duplicate_assignment,
                    "not-an-object",
                ],
            }
        )

        self.assertIsNone(loaded.video_metadata)
        self.assertEqual(loaded.current_frame_index, 0)
        self.assertIsNone(loaded.camera_pose)
        self.assertEqual(loaded.strokes_for_frame(0), [_stroke()])
        self.assertIsNone(loaded.selected_surface_type)
        self.assertEqual(loaded.selected_surface_ids, ())
        self.assertEqual(loaded.assignments, [_assignment()])

    def test_non_object_payload_returns_empty_state(self) -> None:
        self.assertEqual(SurfaceTextureData.from_dict(None), SurfaceTextureData())
        self.assertEqual(SurfaceTextureData.from_dict([]), SurfaceTextureData())


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
