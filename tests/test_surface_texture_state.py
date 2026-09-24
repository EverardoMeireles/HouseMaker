# ### Imports ###
from __future__ import annotations

import unittest
from dataclasses import replace

from housemaker.camera_models import CameraPose
from housemaker.generation_state import MASK_MODE_PAINT, MaskPoint, MaskStroke
from housemaker.pbr_maps import (
    ATLAS_MAP_BASE_COLOR,
    PBR_MAP_METALLIC,
    PBR_MAP_NORMAL,
    PBR_MAP_ROUGHNESS,
    PBR_MAP_TYPES,
)
from housemaker.surface_texture_state import (
    DEFAULT_SURFACE_TEXTURE_RESOLUTION,
    DEFAULT_TEXTURE_REPEAT_SIZE_M,
    SURFACE_PBR_ALIGNMENT_VERSION,
    SURFACE_TEXTURE_RESOLUTIONS,
    SURFACE_TEXTURE_SCHEMA_VERSION,
    SURFACE_TILING_MODE_EDGE_VARIANTS,
    SURFACE_TILING_MODE_NONE,
    SURFACE_TILING_MODE_WHOLE_REPEATS,
    SURFACE_TYPE_CEILING,
    SURFACE_TYPE_FLOOR,
    SURFACE_TYPE_WALL,
    SurfaceTextureAssignment,
    SurfaceTextureData,
    SurfaceTextureVariant,
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

    def test_multi_frame_strokes_are_isolated_and_empty_frames_are_removed(
        self,
    ) -> None:
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
    def test_selection_accepts_exact_geometry_ids_and_deduplicates_in_order(
        self,
    ) -> None:
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

    def test_selection_accepts_typed_stable_edited_face_ids(self) -> None:
        for surface_type, face_token in (
            (SURFACE_TYPE_WALL, "a" * 32),
            (SURFACE_TYPE_FLOOR, "0123456789abcdef" * 2),
            (SURFACE_TYPE_CEILING, "f" * 32),
        ):
            with self.subTest(surface_type=surface_type):
                surface_id = (
                    f"level:2/edit-face:{face_token}:{surface_type}"
                )
                state = SurfaceTextureData(
                    selected_surface_type=None,
                    selected_surface_ids=(surface_id,),
                )

                self.assertEqual(state.selected_surface_type, surface_type)
                self.assertEqual(state.selected_surface_ids, (surface_id,))

    def test_selection_accepts_stable_stair_part_ids(self) -> None:
        stair_id = "0123456789abcdef" * 2
        for surface_type, part_kind in (
            (SURFACE_TYPE_FLOOR, "treads"),
            (SURFACE_TYPE_WALL, "support"),
            (SURFACE_TYPE_WALL, "risers"),
            (SURFACE_TYPE_WALL, "stringers"),
        ):
            with self.subTest(part_kind=part_kind):
                surface_id = (
                    f"stair:{stair_id}/part:{part_kind}:{surface_type}"
                )
                state = SurfaceTextureData(
                    selected_surface_type=None,
                    selected_surface_ids=(surface_id,),
                )

                self.assertEqual(state.selected_surface_type, surface_type)
                self.assertEqual(state.selected_surface_ids, (surface_id,))

    def test_selection_rejects_malformed_stair_part_ids(self) -> None:
        stair_id = "a" * 32
        invalid_surface_ids = (
            f"stair:{'a' * 31}/part:treads:floor",
            f"stair:{'A' * 32}/part:treads:floor",
            f"stair:{stair_id}/part:treads:wall",
            f"stair:{stair_id}/part:unknown:wall",
            f"stair:{stair_id}/part:left_stringer:wall",
            f"stair:{stair_id}/part:right_stringer:wall",
            f"stair:{stair_id}/part:left-stringer:wall",
        )
        for surface_id in invalid_surface_ids:
            with self.subTest(surface_id=surface_id):
                with self.assertRaises(ValueError):
                    SurfaceTextureData(
                        selected_surface_type=SURFACE_TYPE_WALL,
                        selected_surface_ids=(surface_id,),
                    )

    def test_selection_rejects_malformed_edited_face_ids(self) -> None:
        invalid_surface_ids = (
            f"level:2/edit-face:{'a' * 31}:wall",
            f"level:2/edit-face:{'a' * 33}:wall",
            f"level:2/edit-face:{'A' * 32}:wall",
            f"level:2/edit-face:{'a' * 32}:roof",
            f"level:2/room:5/edit-face:{'a' * 32}:wall",
        )
        for surface_id in invalid_surface_ids:
            with self.subTest(surface_id=surface_id):
                with self.assertRaises(ValueError):
                    SurfaceTextureData(
                        selected_surface_type=SURFACE_TYPE_WALL,
                        selected_surface_ids=(surface_id,),
                    )

    def test_selection_rejects_mixed_types_and_malformed_ids(self) -> None:
        invalid_selections = (
            (
                SURFACE_TYPE_WALL,
                ("level:2/room:0/wall:1:2", "level:2/room:0/floor"),
            ),
            (SURFACE_TYPE_CEILING, ("level:2/room:0/floor",)),
            (SURFACE_TYPE_WALL, ("room:0/wall:1:2",)),
            (SURFACE_TYPE_WALL, ("level:-1/room:0/wall:1:2",)),
            (SURFACE_TYPE_WALL, ("level:2/room:0/wall:1:2/overlay:1",)),
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
    def test_tiling_fix_needed_round_trips_and_defaults_for_legacy_data(
        self,
    ) -> None:
        assignment = replace(_assignment(), tiling_fix_needed=True)

        payload = assignment.to_dict()
        self.assertIs(payload["tiling_fix_needed"], True)
        restored = SurfaceTextureAssignment.from_dict(payload)
        payload.pop("tiling_fix_needed")
        legacy = SurfaceTextureAssignment.from_dict(payload)

        self.assertIs(restored.tiling_fix_needed, True)
        self.assertEqual(restored, assignment)
        self.assertIs(legacy.tiling_fix_needed, False)

    def test_tiling_fix_needed_requires_an_exact_boolean(self) -> None:
        for value in (0, 1, None, "true", [], {}):
            with self.subTest(value=value):
                with self.assertRaisesRegex(TypeError, "tiling fix needed"):
                    replace(_assignment(), tiling_fix_needed=value)
                with self.assertRaisesRegex(TypeError, "tiling fix needed"):
                    SurfaceTextureAssignment.from_dict(
                        _assignment().to_dict() | {"tiling_fix_needed": value}
                    )

    def test_texture_repeat_size_round_trips_and_defaults_for_legacy_data(self) -> None:
        assignment = replace(_assignment(), texture_repeat_size_m=0.75)

        restored = SurfaceTextureAssignment.from_dict(assignment.to_dict())
        legacy_payload = assignment.to_dict()
        legacy_payload.pop("texture_repeat_size_m")
        legacy = SurfaceTextureAssignment.from_dict(legacy_payload)

        self.assertEqual(restored.texture_repeat_size_m, 0.75)
        self.assertEqual(restored, assignment)
        self.assertEqual(legacy.texture_repeat_size_m, DEFAULT_TEXTURE_REPEAT_SIZE_M)

    def test_texture_repeat_size_requires_a_positive_finite_number(self) -> None:
        invalid_values = (
            0,
            -0.01,
            float("nan"),
            float("inf"),
            float("-inf"),
            True,
            None,
            "1.0",
        )
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "repeat size"):
                    replace(_assignment(), texture_repeat_size_m=value)
                with self.assertRaisesRegex(ValueError, "repeat size"):
                    SurfaceTextureAssignment.from_dict(
                        _assignment().to_dict() | {"texture_repeat_size_m": value}
                    )

    def test_tiling_mode_and_seed_round_trip(self) -> None:
        for mode in (
            SURFACE_TILING_MODE_NONE,
            SURFACE_TILING_MODE_WHOLE_REPEATS,
            SURFACE_TILING_MODE_EDGE_VARIANTS,
        ):
            with self.subTest(mode=mode):
                assignment = replace(_assignment(), tiling_mode=mode, tiling_seed=42)

                payload = assignment.to_dict()
                restored = SurfaceTextureAssignment.from_dict(payload)

                self.assertEqual(payload["tiling_mode"], mode)
                self.assertEqual(payload["tiling_seed"], 42)
                self.assertEqual(restored, assignment)

    def test_legacy_assignment_defaults_to_no_tiling(self) -> None:
        payload = _assignment().to_dict()
        payload.pop("tiling_mode")
        payload.pop("tiling_seed")

        restored = SurfaceTextureAssignment.from_dict(payload)

        self.assertEqual(restored.tiling_mode, SURFACE_TILING_MODE_NONE)
        self.assertEqual(restored.tiling_seed, 0)

    def test_assignment_rejects_invalid_tiling_configuration(self) -> None:
        for mode, seed in (
            ("random", 0),
            ("Whole_Repeats", 0),
            (None, 0),
            ([], 0),
            (SURFACE_TILING_MODE_WHOLE_REPEATS, -1),
            (SURFACE_TILING_MODE_EDGE_VARIANTS, True),
            (SURFACE_TILING_MODE_NONE, 1.5),
            (SURFACE_TILING_MODE_NONE, "3"),
        ):
            with self.subTest(mode=mode, seed=seed):
                with self.assertRaises(ValueError):
                    replace(_assignment(), tiling_mode=mode, tiling_seed=seed)
                with self.assertRaises(ValueError):
                    SurfaceTextureAssignment.from_dict(
                        _assignment().to_dict()
                        | {"tiling_mode": mode, "tiling_seed": seed}
                    )

    def test_exact_resolution_variants_round_trip_and_infer_active_selection(
        self,
    ) -> None:
        variants = tuple(
            SurfaceTextureVariant(
                resolution=resolution,
                asset_path=f"textures/oak-{resolution}.png",
            )
            for resolution in SURFACE_TEXTURE_RESOLUTIONS
        )
        assignment = SurfaceTextureAssignment(
            assignment_id="oak",
            surface_type=SURFACE_TYPE_FLOOR,
            surface_ids=("level:2/room:0/floor",),
            provider="provider",
            asset_path="textures/oak-1024.png",
            texture_variants=tuple(reversed(variants)),
        )

        restored = SurfaceTextureAssignment.from_dict(assignment.to_dict())

        self.assertEqual(SURFACE_TEXTURE_SCHEMA_VERSION, 11)
        self.assertEqual(
            restored.selected_texture_resolution,
            DEFAULT_SURFACE_TEXTURE_RESOLUTION,
        )
        self.assertEqual(
            (restored.texture_width, restored.texture_height),
            (1024, 1024),
        )
        self.assertEqual(restored.texture_variants, variants)
        self.assertEqual(
            restored.texture_variant_for_resolution(2048),
            variants[-1],
        )
        self.assertIsNone(restored.texture_variant_for_resolution(4096))

    def test_legacy_assignment_without_variants_remains_unchanged(self) -> None:
        legacy = _assignment()

        restored = SurfaceTextureAssignment.from_dict(legacy.to_dict())

        self.assertEqual(restored, legacy)
        self.assertEqual(restored.texture_variants, ())
        self.assertIsNone(restored.selected_texture_resolution)

    def test_assignment_appends_variant_fields_after_legacy_positional_fields(
        self,
    ) -> None:
        assignment = SurfaceTextureAssignment(
            "legacy-positional",
            SURFACE_TYPE_FLOOR,
            ("level:2/room:0/floor",),
            "provider",
            "textures/floor.png",
            "task",
            4.0,
            "Floor",
            (1,),
            512,
            256,
        )

        self.assertEqual(assignment.asset_path, "textures/floor.png")
        self.assertEqual(
            (assignment.texture_width, assignment.texture_height),
            (512, 256),
        )
        self.assertEqual(assignment.texture_variants, ())
        self.assertIsNone(assignment.selected_texture_resolution)

    def test_assignment_rejects_incomplete_or_inconsistent_variants(self) -> None:
        variants = tuple(
            SurfaceTextureVariant(
                resolution=resolution,
                asset_path=f"textures/oak-{resolution}.png",
            )
            for resolution in SURFACE_TEXTURE_RESOLUTIONS
        )
        base_arguments = {
            "assignment_id": "oak",
            "surface_type": SURFACE_TYPE_FLOOR,
            "surface_ids": ("level:2/room:0/floor",),
            "provider": "provider",
            "asset_path": "textures/oak-1024.png",
            "texture_variants": variants,
            "selected_texture_resolution": 1024,
        }
        invalid_overrides = (
            {"texture_variants": variants[:-1]},
            {"texture_variants": (*variants, variants[0])},
            {"asset_path": "textures/oak-512.png"},
            {"texture_width": 512, "texture_height": 512},
            {"selected_texture_resolution": 4096},
        )
        for overrides in invalid_overrides:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    SurfaceTextureAssignment(**(base_arguments | overrides))

    def test_surface_texture_variant_rejects_invalid_resolution_and_path(self) -> None:
        for resolution, asset_path in (
            (256, "textures/oak.png"),
            (True, "textures/oak.png"),
            (512, "../oak.png"),
        ):
            with self.subTest(resolution=resolution, asset_path=asset_path):
                with self.assertRaises(ValueError):
                    SurfaceTextureVariant(resolution, asset_path)

    def test_pbr_map_paths_and_selection_round_trip(self) -> None:
        variants = tuple(
            SurfaceTextureVariant(
                resolution=resolution,
                asset_path=f"textures/stone-{resolution}.png",
                map_asset_paths={
                    ATLAS_MAP_BASE_COLOR: f"textures/stone-{resolution}.png",
                    PBR_MAP_NORMAL: f"textures/stone-{resolution}.normal.png",
                    PBR_MAP_ROUGHNESS: (
                        f"textures/stone-{resolution}.roughness.png"
                    ),
                    PBR_MAP_METALLIC: (
                        f"textures/stone-{resolution}.metallic.png"
                    ),
                },
            )
            for resolution in SURFACE_TEXTURE_RESOLUTIONS
        )
        assignment = SurfaceTextureAssignment(
            assignment_id="stone",
            surface_type=SURFACE_TYPE_WALL,
            surface_ids=("level:2/room:0/wall:1:2",),
            provider="meshy",
            asset_path="textures/stone-1024.png",
            provider_task_id="image-task",
            provider_pbr_task_id="pbr-task",
            texture_variants=variants,
            enabled_pbr_maps=(PBR_MAP_METALLIC, PBR_MAP_NORMAL),
            pbr_alignment_version=SURFACE_PBR_ALIGNMENT_VERSION,
        )

        restored = SurfaceTextureAssignment.from_dict(assignment.to_dict())

        self.assertEqual(
            restored.enabled_pbr_maps,
            (PBR_MAP_NORMAL, PBR_MAP_METALLIC),
        )
        self.assertEqual(restored.available_pbr_maps, PBR_MAP_TYPES)
        self.assertEqual(restored.provider_pbr_task_id, "pbr-task")
        self.assertEqual(
            restored.pbr_alignment_version,
            SURFACE_PBR_ALIGNMENT_VERSION,
        )
        normal_path = restored.texture_variants[1].asset_path_for_map(
            PBR_MAP_NORMAL
        )
        self.assertEqual(normal_path, "textures/stone-1024.normal.png")

    def test_pbr_map_paths_are_safe_and_consistent_across_resolutions(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsafe"):
            SurfaceTextureVariant(
                512,
                "textures/base.png",
                {PBR_MAP_NORMAL: "../normal.png"},
            )

        variants = (
            SurfaceTextureVariant(
                512,
                "textures/base-512.png",
                {PBR_MAP_NORMAL: "textures/normal-512.png"},
            ),
            SurfaceTextureVariant(1024, "textures/base-1024.png"),
            SurfaceTextureVariant(
                2048,
                "textures/base-2048.png",
                {PBR_MAP_NORMAL: "textures/normal-2048.png"},
            ),
        )
        with self.assertRaisesRegex(ValueError, "same maps"):
            SurfaceTextureAssignment(
                assignment_id="stone",
                surface_type=SURFACE_TYPE_FLOOR,
                surface_ids=("level:2/room:0/floor",),
                provider="meshy",
                asset_path="textures/base-1024.png",
                texture_variants=variants,
            )

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
        self.assertEqual(
            (assignment.texture_width, assignment.texture_height),
            (512, 256),
        )

    def test_unused_assignment_round_trips_with_no_surface_ids(self) -> None:
        assignment = SurfaceTextureAssignment(
            assignment_id="unused-stone",
            surface_type=SURFACE_TYPE_FLOOR,
            surface_ids=(),
            provider="meshy",
            asset_path="textures/unused-stone.png",
        )

        restored = SurfaceTextureData.from_dict(
            SurfaceTextureData(assignments=[assignment]).to_dict()
        )

        self.assertEqual(len(restored.assignments), 1)
        self.assertEqual(restored.assignments[0].assignment_id, "unused-stone")
        self.assertEqual(restored.assignments[0].surface_ids, ())

    def test_assignment_rejects_unsafe_assets_invalid_area_and_partial_dimensions(
        self,
    ) -> None:
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
    def test_schema_four_overlay_and_inpaint_data_is_ignored(self) -> None:
        wall_id = "level:2/room:0/wall:1:2"
        overlay_id = f"{wall_id}/overlay:1"
        mixed_assignment = _assignment().to_dict() | {
            "assignment_id": "mixed",
            "surface_ids": [wall_id, overlay_id],
        }
        overlay_assignment = _assignment().to_dict() | {
            "assignment_id": "overlay-only",
            "surface_ids": [overlay_id],
        }
        loaded = SurfaceTextureData.from_dict(
            {
                "schema_version": 4,
                "selected_surface_type": SURFACE_TYPE_WALL,
                "selected_surface_ids": [overlay_id, wall_id],
                "texture_mask_strokes": {
                    overlay_id: [_stroke(0.2).to_dict()],
                    wall_id: [_stroke(0.8).to_dict()],
                },
                "overlay_planes": [
                    {
                        "surface_id": overlay_id,
                        "parent_surface_id": wall_id,
                        "normal_offset_meters": 0.003,
                    }
                ],
                "assignments": [mixed_assignment, overlay_assignment],
                "localized_inpaint_undo_stack": [
                    {
                        "previous_assignments": [overlay_assignment],
                        "replacement_assignment_ids": ["overlay-after"],
                        "affected_surface_ids": [overlay_id],
                        "previous_texture_mask_strokes": {
                            overlay_id: [_stroke().to_dict()]
                        },
                    }
                ],
            }
        )

        self.assertEqual(loaded.selected_surface_ids, (wall_id,))
        self.assertEqual(len(loaded.assignments), 1)
        self.assertEqual(loaded.assignments[0].assignment_id, "mixed")
        self.assertEqual(loaded.assignments[0].surface_ids, (wall_id,))
        self.assertFalse(hasattr(loaded, "texture_mask_strokes"))
        self.assertFalse(hasattr(loaded, "localized_inpaint_undo_stack"))
        self.assertFalse(hasattr(loaded, "overlay_planes"))
        self.assertNotIn("overlay_planes", loaded.to_dict())
        self.assertNotIn("texture_mask_strokes", loaded.to_dict())
        self.assertNotIn("localized_inpaint_undo_stack", loaded.to_dict())

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

    def test_malformed_optional_records_fall_back_without_losing_valid_records(
        self,
    ) -> None:
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
