"""Reusable Surface textures stay unpacked until assigned to live geometry."""

# ### Environment and imports ###
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from housemaker.surface_geometry import build_fixed_surfaces
from housemaker.surface_texture_state import SurfaceTextureData
from housemaker.texture_atlas_state import (
    ATLAS_TYPE_OBJECT,
    ATLAS_TYPE_SURFACE,
    TextureAtlasData,
)
from housemaker.texture_atlas_workspace import build_atlas_wall_texture_source_id
from tests import test_texture_atlas_main_integration as fixtures


# ### Actual Main workspace with temporary settings and image assets ###
class SurfaceTextureLibraryMainIntegrationTests(unittest.TestCase):
    setUp = fixtures.TextureAtlasMainIntegrationTests.setUp
    tearDown = fixtures.TextureAtlasMainIntegrationTests.tearDown

    def _install_room(self) -> dict[str, str]:
        fixtures._add_square_room_to_level(self.workspace.current_level)
        self.workspace.surface_texture_generation.set_levels(self.workspace.levels)
        surfaces = tuple(build_fixed_surfaces(self.workspace.levels))
        self.workspace._set_canvas_viewer_targets(surfaces)
        return {
            kind: next(
                surface.surface_id
                for surface in surfaces
                if surface.surface_type == kind
            )
            for kind in ("wall", "floor", "ceiling")
        }

    def _assignment(
        self,
        assignment_id="material",
        *,
        surface_type="surface",
        surface_ids=(),
        resolution=512,
    ):
        return fixtures._wall_texture_assignment_with_variants(
            self.settings.path.parent / "surface_textures",
            assignment_id=assignment_id,
            surface_type=surface_type,
            surface_ids=surface_ids,
            selected_resolution=resolution,
        )

    def _load_assignments(self, *assignments) -> None:
        data = SurfaceTextureData(assignments=list(assignments))
        self.workspace.surface_texture_generation.set_data(data)
        self.workspace._atlas_generation_signature = None
        self.workspace._sync_atlas_object_texture_sources(
            automatically_assign_scene_textures=False
        )

    # ### Library inventory is independent of exported scene/Atlas membership ###
    def test_generic_library_entry_is_listed_but_not_bound_or_automatically_packed(
        self,
    ) -> None:
        assignment = self._assignment()
        self._load_assignments(assignment)
        self.workspace.surface_texture_generation.generation_completed.emit(assignment)
        self.workspace._refresh_scene_atlas_texture_requirements()

        source_id = build_atlas_wall_texture_source_id(assignment.assignment_id)
        atlas = self.workspace.texture_atlas_workspace
        self.assertIn(source_id, atlas._sources_by_object_id)
        self.assertIn(source_id, atlas._surface_texture_entries_by_id)
        self.assertEqual(
            atlas._surface_texture_entries_by_id[source_id].surface_type, "surface"
        )
        self.assertEqual(atlas.surface_list.count(), 1)
        self.assertFalse(atlas.is_source_assigned_to_any_atlas(source_id))
        self.assertEqual(atlas.get_data().atlases, [])
        self.assertEqual(self.workspace._build_bound_surface_atlas_source_ids(), ())
        self.assertNotIn(
            source_id, self.workspace._build_required_scene_atlas_source_ids()
        )
        self.assertFalse(self.workspace._pending_surface_texture_ids)
        self.assertFalse(self.workspace._surface_texture_mesh_update_timer.isActive())

    def test_assign_without_a_live_target_keeps_library_material_unpacked(self) -> None:
        assignment = self._assignment()
        self._load_assignments(assignment)
        source_id = build_atlas_wall_texture_source_id(assignment.assignment_id)

        self.workspace._handle_atlas_surface_assign_requested(source_id)

        atlas = self.workspace.texture_atlas_workspace
        self.assertFalse(atlas.is_source_assigned_to_any_atlas(source_id))
        self.assertEqual(
            self.workspace.surface_texture_generation.get_assignment(
                assignment.assignment_id
            ).surface_ids,
            (),
        )
        self.assertIn("Select at least one", atlas.status_label.text())

    # ### Ordinary material families cross wall/floor/ceiling boundaries ###
    def test_generic_assign_can_target_walls_floors_and_ceilings_together(self) -> None:
        targets = tuple(self._install_room().values())
        self.workspace._set_mesh_edit_update_delay_seconds(60.0)
        assignment = self._assignment()
        self._load_assignments(assignment)
        source_id = build_atlas_wall_texture_source_id(assignment.assignment_id)
        self.workspace._atlas_surface_assignment_target_ids = targets

        with (
            patch.object(self.workspace.texture_atlas_workspace, "_materialize_atlas"),
            patch.object(self.workspace, "_queue_viewer_preview_refresh") as refresh,
        ):
            self.workspace._handle_atlas_surface_assign_requested(source_id)
            fixtures._qt_application.processEvents()

            self.assertEqual(self.workspace._pending_surface_texture_ids, set(targets))
            self.assertEqual(
                set(self.workspace.viewer.get_pending_surface_texture_preview_ids()),
                set(targets),
            )
            self.assertTrue(
                self.workspace._surface_texture_mesh_update_timer.isActive()
            )
            self.assertTrue(self.workspace.confirm_mesh_updates_button.isEnabled())
            refresh.assert_not_called()

            self.workspace._commit_pending_surface_texture_mesh_update()

            self.assertFalse(self.workspace._pending_surface_texture_ids)
            self.assertFalse(
                self.workspace._surface_texture_mesh_update_timer.isActive()
            )
            self.assertEqual(
                set(self.workspace._committing_surface_texture_revisions), set(targets)
            )
            self.assertEqual(
                set(self.workspace.viewer.get_pending_surface_texture_preview_ids()),
                set(targets),
            )
            refresh.assert_called_once()

            self.workspace._canvas_viewer_preview_revision = (
                self.workspace._viewer_preview_revision - 1
            )
            self.workspace._clear_committed_surface_texture_previews_if_displayed()
            self.assertEqual(
                set(self.workspace.viewer.get_pending_surface_texture_preview_ids()),
                set(targets),
            )
            self.workspace._canvas_viewer_preview_revision = (
                self.workspace._viewer_preview_revision
            )
            self.workspace._clear_committed_surface_texture_previews_if_displayed()
            self.assertFalse(self.workspace._committing_surface_texture_revisions)
            self.assertEqual(
                self.workspace.viewer.get_pending_surface_texture_preview_ids(), ()
            )

        updated = self.workspace.surface_texture_generation.get_assignment(
            assignment.assignment_id
        )
        self.assertEqual(set(updated.surface_ids), set(targets))
        self.assertEqual(updated.surface_type, "surface")
        atlas = self.workspace.texture_atlas_workspace.selected_atlas
        self.assertIsNotNone(atlas)
        self.assertEqual(atlas.atlas_type, ATLAS_TYPE_SURFACE)
        self.assertTrue(atlas.name.startswith("[SURFACE]"))
        self.assertIsNotNone(atlas.placement_for_object(source_id))
        self.assertIn(source_id, self.workspace._build_bound_surface_atlas_source_ids())

    def test_failed_or_rejected_assign_leaves_no_pending_texture_update(self) -> None:
        targets = tuple(self._install_room().values())
        self.workspace._set_mesh_edit_update_delay_seconds(60.0)
        assignment = self._assignment()
        self._load_assignments(assignment)
        source_id = build_atlas_wall_texture_source_id(assignment.assignment_id)
        for failure in ("materialize", "binding"):
            with self.subTest(failure=failure):
                self.workspace._atlas_surface_assignment_target_ids = targets
                with (
                    patch.object(
                        self.workspace.texture_atlas_workspace,
                        "_materialize_atlas",
                        side_effect=(
                            OSError("Atlas write failed")
                            if failure == "materialize"
                            else None
                        ),
                    ),
                    patch.object(
                        self.workspace.surface_texture_generation,
                        "apply_assignment_texture",
                        return_value=False,
                    ) as apply,
                    patch.object(
                        self.workspace, "_queue_viewer_preview_refresh"
                    ) as refresh,
                ):
                    self.workspace._handle_atlas_surface_assign_requested(source_id)

                if failure == "materialize":
                    apply.assert_not_called()
                else:
                    apply.assert_called_once()
                refresh.assert_not_called()
                self.assertFalse(self.workspace._pending_surface_texture_ids)
                self.assertFalse(self.workspace._committing_surface_texture_revisions)
                self.assertFalse(
                    self.workspace._surface_texture_mesh_update_timer.isActive()
                )
                self.assertEqual(
                    self.workspace.viewer.get_pending_surface_texture_preview_ids(), ()
                )
                self.assertEqual(
                    self.workspace.surface_texture_generation.get_assignment(
                        assignment.assignment_id
                    ).surface_ids,
                    (),
                )
                self.assertFalse(
                    self.workspace.texture_atlas_workspace.is_source_assigned_to_any_atlas(
                        source_id
                    )
                )

    def test_existing_ordinary_texture_types_accept_other_ordinary_surfaces(
        self,
    ) -> None:
        targets = tuple(self._install_room().values())
        assignments = tuple(
            self._assignment(f"material-{kind}", surface_type=kind)
            for kind in ("wall", "floor", "ceiling")
        )
        self._load_assignments(*assignments)
        for assignment in assignments:
            with self.subTest(surface_type=assignment.surface_type):
                self.assertTrue(
                    self.workspace._atlas_surface_targets_match_assignment(
                        assignment, targets
                    )
                )
                source_id = build_atlas_wall_texture_source_id(assignment.assignment_id)
                self.workspace._atlas_surface_assignment_target_ids = targets
                with patch.object(
                    self.workspace.texture_atlas_workspace, "_materialize_atlas"
                ):
                    self.workspace._handle_atlas_surface_assign_requested(source_id)
                updated = self.workspace.surface_texture_generation.get_assignment(
                    assignment.assignment_id
                )
                self.assertEqual(set(updated.surface_ids), set(targets))
                self.assertTrue(
                    self.workspace.texture_atlas_workspace.is_source_assigned_to_any_atlas(
                        source_id
                    )
                )

    # ### Assign automatically overflows only Surface Atlases ###
    def test_generic_assign_creates_surface_overflow_even_when_automatic_creation_is_disabled(
        self,
    ) -> None:
        target_ids = self._install_room()
        assignment = self._assignment()
        filler = self._assignment(
            "filler", surface_ids=(target_ids["floor"],), resolution=2048
        )
        self._load_assignments(assignment, filler)
        atlas_workspace = self.workspace.texture_atlas_workspace
        filler_id = build_atlas_wall_texture_source_id(filler.assignment_id)
        filler_source = atlas_workspace._sources_by_object_id[filler_id]
        data = TextureAtlasData()
        full = data.create_atlas(
            "Full", 2048, atlas_id="surfaces", atlas_type=ATLAS_TYPE_SURFACE
        )
        data.assign_object(
            full.atlas_id,
            filler_id,
            filler_source.texture_path,
            filler_source.texture_resolution,
        )
        objects = data.create_atlas(
            "Furniture", 2048, atlas_id="objects", atlas_type=ATLAS_TYPE_OBJECT
        )
        data.select_atlas(objects.atlas_id)
        atlas_workspace.set_data(data)
        self.assertFalse(
            self.workspace.settings_widget.get_settings().automatic_atlas_creation
        )
        self.workspace._atlas_surface_assignment_target_ids = (target_ids["wall"],)
        source_id = build_atlas_wall_texture_source_id(assignment.assignment_id)

        with patch.object(atlas_workspace, "_materialize_atlas"):
            self.workspace._handle_atlas_surface_assign_requested(source_id)

        updated = atlas_workspace.get_data()
        self.assertEqual(len(updated.atlases), 3)
        self.assertEqual(updated.atlas_by_id(objects.atlas_id).placements, [])
        self.assertIsNotNone(
            updated.atlas_by_id(full.atlas_id).placement_for_object(filler_id)
        )
        overflow = next(
            atlas
            for atlas in updated.atlases
            if atlas.placement_for_object(source_id) is not None
        )
        self.assertNotEqual(overflow.atlas_id, full.atlas_id)
        self.assertEqual(overflow.atlas_type, ATLAS_TYPE_SURFACE)
        self.assertTrue(overflow.name.startswith("[SURFACE]"))


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
