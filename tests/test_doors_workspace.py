# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pyqtgraph.opengl as gl
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from housemaker.door_geometry import (
    assemble_door_model,
    assemble_window_model,
    build_default_door_slot_models,
    build_door_body_model,
)
from housemaker.door_state import (
    DOOR_SLOT_BACK_BODY,
    DOOR_SLOT_BACK_KNOB,
    DOOR_SLOT_BODY,
    DOOR_SLOT_KNOB,
    DOOR_SLOT_LEFT_BODY,
    DOOR_SLOT_LEFT_KNOB,
    DoorDefinition,
    DoorLibraryData,
    DoorPlacement,
    DoorSlotData,
    GlassAreaData,
    WindowDefinition,
    WindowPlacement,
)
from housemaker.doors_workspace import (
    ASSET_KIND_DOOR,
    ASSET_KIND_WINDOW,
    OPENING_PREVIEW_FRONT_ARROW_ROLE,
    DoorsWorkspace,
)
from housemaker.viewer import GlbViewerWidget

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Test fixtures ###
def _door(
    door_id: str = "door-1",
    *,
    name: str = "New door 1",
    shape: str = "rectangular",
    arch_amount: float = 0.0,
) -> DoorDefinition:
    return DoorDefinition(
        door_id=door_id,
        name=name,
        source_doorway_id=f"doorway-{door_id}",
        width_meters=0.9,
        height_meters=2.1,
        thickness_meters=0.04,
        shape=shape,
        arch_amount=arch_amount,
        slots=(
            DoorSlotData(
                slot_id=DOOR_SLOT_BODY,
                source_object_id=f"{door_id}-body",
                joined=True,
            ),
            DoorSlotData(
                slot_id=DOOR_SLOT_KNOB,
                source_object_id=f"{door_id}-knob",
            ),
        ),
    )


def _window(
    window_id: str = "window-1",
    *,
    name: str = "New window 1",
    glass_areas: tuple[GlassAreaData, ...] = (),
) -> WindowDefinition:
    return WindowDefinition(
        window_id=window_id,
        name=name,
        source_window_frame_id=f"window-frame-{window_id}",
        width_meters=1.2,
        height_meters=1.0,
        thickness_meters=0.04,
        slots=(
            DoorSlotData(
                slot_id=DOOR_SLOT_BODY,
                source_object_id=f"{window_id}-body",
                joined=True,
            ),
            DoorSlotData(
                slot_id=DOOR_SLOT_KNOB,
                source_object_id=f"{window_id}-handle",
            ),
        ),
        glass_areas=glass_areas,
    )


def _complete_door_model(door: DoorDefinition):
    """Build generated preview meshes for every editable door slot."""

    slot_models = build_default_door_slot_models(door)
    for slot in door.slots:
        if slot.slot_id not in {DOOR_SLOT_KNOB, DOOR_SLOT_BACK_KNOB}:
            continue
        slot_models[slot.source_object_id] = build_door_body_model(
            0.08,
            0.04,
            0.12,
            shape="rectangular",
            arch_amount=0.0,
            geometry_name=f"generated_{slot.slot_id}_fixture",
        )
    return assemble_door_model(
        door,
        slot_models,
        include_unjoined=True,
    )


def _preview_face_geometry(
    model: object,
    object_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten one placed preview part into the viewer's coordinate space."""

    preview = next(
        item
        for item in model.preview_placed_objects
        if item.object_id == object_id
    )
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    vertex_offset = 0
    for mesh in preview.meshes:
        local_vertices = np.asarray(mesh.vertices, dtype=float)
        homogeneous = np.column_stack(
            (local_vertices, np.ones(len(local_vertices), dtype=float))
        )
        transformed = (
            np.asarray(preview.placement_transform, dtype=float)
            @ homogeneous.T
        ).T[:, :3]
        mesh_faces = np.asarray(mesh.faces, dtype=np.int64)
        vertices.append(transformed)
        faces.append(mesh_faces + vertex_offset)
        vertex_offset += len(local_vertices)
    return np.concatenate(vertices), np.concatenate(faces)


# ### Doors workspace tests ###
class DoorsWorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = DoorsWorkspace()

    def tearDown(self) -> None:
        self.workspace.close()
        self.workspace.deleteLater()
        _qt_application.processEvents()

    # ### Tree presentation ###
    def test_tree_shows_each_door_and_the_two_ordered_slots(self) -> None:
        door = _door(shape="arch", arch_amount=0.35)

        self.workspace.set_data(DoorLibraryData(doors=[door]))

        self.assertEqual(self.workspace.door_tree.columnCount(), 1)
        self.assertEqual(
            self.workspace.door_tree.headerItem().text(0),
            "Door, window / part",
        )
        self.assertEqual(self.workspace.door_tree.topLevelItemCount(), 1)
        door_item = self.workspace.door_tree.topLevelItem(0)
        self.assertEqual(door_item.text(0), "New door 1")
        self.assertEqual(door_item.childCount(), 2)
        self.assertEqual(
            [door_item.child(index).text(0) for index in range(2)],
            ["Body", "Handle"],
        )

    def test_profile_section_and_its_legacy_text_labels_are_removed(self) -> None:
        self.assertFalse(hasattr(self.workspace, "profile_group"))
        self.assertFalse(hasattr(self.workspace, "profile_name_label"))
        self.assertFalse(hasattr(self.workspace, "profile_shape_label"))
        self.assertFalse(hasattr(self.workspace, "profile_dimensions_label"))
        self.assertFalse(hasattr(self.workspace, "profile_source_label"))

    def test_tree_shows_windows_with_body_and_handle_slots(self) -> None:
        door = _door()
        window = _window()

        self.workspace.set_data(
            DoorLibraryData(doors=[door], windows=[window])
        )

        self.assertEqual(self.workspace.door_tree.topLevelItemCount(), 2)
        window_item = self.workspace.door_tree.topLevelItem(1)
        self.assertEqual(window_item.text(0), window.name)
        self.assertEqual(
            [window_item.child(index).text(0) for index in range(2)],
            ["Body", "Handle"],
        )
        self.assertTrue(
            self.workspace.select_window_slot(
                window.window_id,
                DOOR_SLOT_KNOB,
            )
        )
        self.assertEqual(self.workspace.selected_asset_kind(), ASSET_KIND_WINDOW)
        self.assertEqual(self.workspace.selected_window_id(), window.window_id)
        self.assertEqual(self.workspace.selected_door_id(), None)
        self.assertEqual(
            self.workspace.selected_window_slot(),
            window.get_slot(DOOR_SLOT_KNOB),
        )
        self.assertTrue(self.workspace.make_double_checkbox.isEnabled())
        self.assertTrue(self.workspace.make_double_sided_checkbox.isEnabled())

    def test_window_selection_and_slot_requests_are_asset_aware(self) -> None:
        window = _window()
        selections: list[tuple[object, ...]] = []
        edit_requests: list[tuple[object, ...]] = []
        legacy_requests: list[tuple[object, ...]] = []
        self.workspace.asset_selection_changed.connect(
            lambda *values: selections.append(values)
        )
        self.workspace.asset_slot_edit_requested.connect(
            lambda *values: edit_requests.append(values)
        )
        self.workspace.slot_edit_requested.connect(
            lambda *values: legacy_requests.append(values)
        )
        self.workspace.set_data(DoorLibraryData(windows=[window]))
        window_item = self.workspace.door_tree.topLevelItem(0)
        handle_item = window_item.child(1)

        self.workspace.door_tree.itemDoubleClicked.emit(handle_item, 0)

        self.assertEqual(
            edit_requests,
            [(ASSET_KIND_WINDOW, window.window_id, DOOR_SLOT_KNOB)],
        )
        self.assertEqual(legacy_requests, [])
        self.assertIn(
            (
                ASSET_KIND_WINDOW,
                window.window_id,
                DOOR_SLOT_BODY,
                None,
            ),
            selections,
        )

    def test_delete_button_requests_the_selected_asset_owner(self) -> None:
        door = _door()
        window = _window()
        original_data = DoorLibraryData(doors=[door], windows=[window])
        requests: list[tuple[str, str]] = []
        self.workspace.asset_delete_requested.connect(
            lambda *values: requests.append(values)
        )

        self.assertEqual(self.workspace.delete_asset_button.text(), "Delete")
        self.assertEqual(
            self.workspace.delete_asset_button.objectName(),
            "delete_opening_asset_button",
        )
        self.assertFalse(self.workspace.delete_asset_button.isEnabled())
        self.workspace.set_data(original_data)
        self.assertTrue(
            self.workspace.select_window_slot(
                window.window_id,
                DOOR_SLOT_KNOB,
            )
        )

        self.workspace.delete_asset_button.click()

        self.assertEqual(
            requests,
            [(ASSET_KIND_WINDOW, window.window_id)],
        )
        self.assertEqual(self.workspace.data(), original_data)
        self.workspace.set_data(DoorLibraryData())
        self.assertFalse(self.workspace.delete_asset_button.isEnabled())

    def test_flip_mesh_button_requires_slot_and_emits_asset_aware_request(
        self,
    ) -> None:
        """Route explicit door/window slot flips without mutating local data."""

        door = _door()
        window = _window()
        original_data = DoorLibraryData(doors=[door], windows=[window])
        requests: list[tuple[str, str, str]] = []
        self.workspace.asset_slot_horizontal_flip_requested.connect(
            lambda *values: requests.append(values)
        )

        self.assertEqual(
            self.workspace.flip_mesh_horizontally_button.text(),
            "Flip mesh horizontally",
        )
        self.assertEqual(
            self.workspace.flip_mesh_horizontally_button.objectName(),
            "flip_opening_slot_mesh_horizontally_button",
        )
        self.assertFalse(
            self.workspace.flip_mesh_horizontally_button.isEnabled()
        )

        self.workspace.set_data(original_data)
        self.assertTrue(
            self.workspace.select_door_slot(door.door_id, DOOR_SLOT_BODY)
        )
        self.assertTrue(self.workspace.flip_mesh_horizontally_button.isEnabled())
        self.workspace.flip_mesh_horizontally_button.click()

        self.assertTrue(
            self.workspace.select_window_slot(
                window.window_id,
                DOOR_SLOT_KNOB,
            )
        )
        self.assertTrue(self.workspace.flip_mesh_horizontally_button.isEnabled())
        self.workspace.flip_mesh_horizontally_button.click()

        self.assertEqual(
            requests,
            [
                (ASSET_KIND_DOOR, door.door_id, DOOR_SLOT_BODY),
                (ASSET_KIND_WINDOW, window.window_id, DOOR_SLOT_KNOB),
            ],
        )
        self.assertEqual(self.workspace.data(), original_data)

    def test_window_double_controls_emit_only_asset_aware_requests(self) -> None:
        window = _window()
        double_requests: list[tuple[object, ...]] = []
        sided_requests: list[tuple[object, ...]] = []
        legacy_double_requests: list[tuple[object, ...]] = []
        legacy_sided_requests: list[tuple[object, ...]] = []
        self.workspace.asset_make_double_changed.connect(
            lambda *values: double_requests.append(values)
        )
        self.workspace.asset_make_double_sided_changed.connect(
            lambda *values: sided_requests.append(values)
        )
        self.workspace.make_double_changed.connect(
            lambda *values: legacy_double_requests.append(values)
        )
        self.workspace.make_double_sided_changed.connect(
            lambda *values: legacy_sided_requests.append(values)
        )
        self.workspace.set_data(DoorLibraryData(windows=[window]))

        self.workspace.make_double_checkbox.click()
        self.assertEqual(
            double_requests,
            [(ASSET_KIND_WINDOW, window.window_id, True)],
        )

        doubled = window.with_double(True)
        self.workspace.upsert_window(doubled, select_body=True)
        self.workspace.make_double_sided_checkbox.click()
        self.assertEqual(
            sided_requests,
            [(ASSET_KIND_WINDOW, window.window_id, True)],
        )
        self.assertEqual(legacy_double_requests, [])
        self.assertEqual(legacy_sided_requests, [])

    def test_double_sided_double_window_groups_all_slots_by_face(self) -> None:
        window = _window().with_double(True).with_double_sided(True)
        self.workspace.set_data(DoorLibraryData(windows=[window]))

        window_item = self.workspace.door_tree.topLevelItem(0)
        self.assertEqual(window_item.childCount(), 2)
        front_item = window_item.child(0)
        back_item = window_item.child(1)
        self.assertEqual(front_item.text(0), "Front")
        self.assertEqual(back_item.text(0), "Back")
        self.assertEqual(
            [front_item.child(index).text(0) for index in range(4)],
            ["Body right", "Body left", "Handle right", "Handle left"],
        )
        self.assertEqual(
            [back_item.child(index).text(0) for index in range(4)],
            ["Body right", "Body left", "Handle right", "Handle left"],
        )
        self.assertTrue(self.workspace.make_double_checkbox.isChecked())
        self.assertTrue(self.workspace.make_double_sided_checkbox.isChecked())

    def test_window_ctrl_copy_and_paste_emits_asset_counterpart_request(
        self,
    ) -> None:
        window = _window().with_double(True)
        asset_requests: list[tuple[object, ...]] = []
        legacy_requests: list[tuple[object, ...]] = []
        self.workspace.asset_slot_mesh_paste_requested.connect(
            lambda *values: asset_requests.append(values)
        )
        self.workspace.slot_mesh_paste_requested.connect(
            lambda *values: legacy_requests.append(values)
        )
        self.workspace.set_data(DoorLibraryData(windows=[window]))

        self.workspace.select_window_slot(window.window_id, DOOR_SLOT_BODY)
        self.workspace.copy_slot_shortcut.activated.emit()
        self.workspace.select_window_slot(window.window_id, DOOR_SLOT_LEFT_BODY)
        self.workspace.paste_slot_shortcut.activated.emit()

        self.assertEqual(
            asset_requests,
            [
                (
                    ASSET_KIND_WINDOW,
                    window.window_id,
                    DOOR_SLOT_BODY,
                    DOOR_SLOT_LEFT_BODY,
                )
            ],
        )
        self.assertEqual(legacy_requests, [])

    def test_window_upsert_and_remove_preserve_door_library_data(self) -> None:
        door = _door()
        window = _window()
        door_placement = DoorPlacement(
            placement_id="door-placement",
            door_id=door.door_id,
            level_index=0,
            doorway_id="doorway-1",
        )
        window_placement = WindowPlacement(
            placement_id="window-placement",
            window_id=window.window_id,
            level_index=0,
            window_frame_id="window-frame-1",
        )
        self.workspace.set_data(
            DoorLibraryData(
                doors=[door],
                placements=[door_placement],
                windows=[window],
                window_placements=[window_placement],
            )
        )

        self.workspace.upsert_window(
            replace(window, name="Kitchen window"),
            select_body=True,
        )

        self.assertEqual(self.workspace.selected_window_id(), window.window_id)
        self.assertEqual(self.workspace.selected_slot_id(), DOOR_SLOT_BODY)
        self.assertEqual(self.workspace.data().doors, [door])
        self.assertEqual(self.workspace.data().placements, [door_placement])
        self.assertTrue(self.workspace.remove_window(window.window_id))
        self.assertEqual(self.workspace.data().doors, [door])
        self.assertEqual(self.workspace.data().placements, [door_placement])
        self.assertEqual(self.workspace.data().windows, [])
        self.assertEqual(self.workspace.data().window_placements, [])

    def test_glass_area_leaf_selects_without_opening_generation(self) -> None:
        glass_area = GlassAreaData(
            glass_area_id="glass-1",
            source_object_id="window-1-glass-1",
            name="Glass area 1",
            minimum_x_ratio=0.2,
            maximum_x_ratio=0.8,
            minimum_z_ratio=0.1,
            maximum_z_ratio=0.9,
        )
        window = _window(glass_areas=(glass_area,))
        edit_requests: list[tuple[object, ...]] = []
        glass_requests: list[tuple[object, ...]] = []
        self.workspace.asset_slot_edit_requested.connect(
            lambda *values: edit_requests.append(values)
        )
        self.workspace.glass_area_creation_requested.connect(
            lambda *values: glass_requests.append(values)
        )
        self.workspace.set_data(DoorLibraryData(windows=[window]))
        window_item = self.workspace.door_tree.topLevelItem(0)
        glass_item = window_item.child(2)

        self.workspace.door_tree.setCurrentItem(glass_item)
        self.workspace.door_tree.itemDoubleClicked.emit(glass_item, 0)

        self.assertEqual(self.workspace.selected_glass_area(), glass_area)
        self.assertEqual(self.workspace.selected_asset_slot(), None)
        self.assertEqual(edit_requests, [])
        self.assertFalse(
            self.workspace.confirm_slot_position_button.isEnabled()
        )
        self.workspace.add_glass_area_button.click()
        self.assertEqual(
            glass_requests,
            [
                (
                    ASSET_KIND_WINDOW,
                    window.window_id,
                    window.get_slot(DOOR_SLOT_BODY).source_object_id,
                )
            ],
        )

    def test_double_sided_door_groups_front_and_back_slots(self) -> None:
        door = _door().with_double_sided(True)
        requests: list[tuple[str, str]] = []
        self.workspace.slot_edit_requested.connect(
            lambda door_id, slot_id: requests.append((door_id, slot_id))
        )

        self.workspace.set_data(DoorLibraryData(doors=[door]))

        door_item = self.workspace.door_tree.topLevelItem(0)
        self.assertEqual(door_item.childCount(), 2)
        front_item = door_item.child(0)
        back_item = door_item.child(1)
        self.assertEqual(front_item.text(0), "Front")
        self.assertEqual(back_item.text(0), "Back")
        self.assertTrue(front_item.isExpanded())
        self.assertTrue(back_item.isExpanded())
        self.assertEqual(
            [front_item.child(index).text(0) for index in range(2)],
            ["Body", "Handle"],
        )
        self.assertEqual(
            [back_item.child(index).text(0) for index in range(2)],
            ["Body", "Handle"],
        )
        self.assertTrue(self.workspace.make_double_sided_checkbox.isChecked())
        self.assertTrue(
            self.workspace.select_door_slot(
                door.door_id,
                DOOR_SLOT_BACK_KNOB,
            )
        )
        self.assertEqual(
            self.workspace.selected_slot_id(),
            DOOR_SLOT_BACK_KNOB,
        )
        self.workspace.door_tree.itemDoubleClicked.emit(back_item.child(0), 0)
        self.assertEqual(requests, [(door.door_id, DOOR_SLOT_BACK_BODY)])

    def test_make_double_sided_checkbox_requests_a_model_transition(self) -> None:
        door = _door()
        requests: list[tuple[str, bool]] = []
        self.workspace.make_double_sided_changed.connect(
            lambda door_id, checked: requests.append((door_id, checked))
        )
        self.workspace.set_data(DoorLibraryData(doors=[door]))

        self.workspace.make_double_sided_checkbox.click()

        self.assertEqual(requests, [(door.door_id, True)])

        updated = door.with_double_sided(True)
        self.workspace.upsert_door(updated)
        self.workspace.make_double_sided_checkbox.click()

        self.assertEqual(
            requests,
            [(door.door_id, True), (door.door_id, False)],
        )

    def test_make_double_shows_contextual_left_and_right_slots(self) -> None:
        door = _door().with_double(True)

        self.workspace.set_data(DoorLibraryData(doors=[door]))

        door_item = self.workspace.door_tree.topLevelItem(0)
        self.assertEqual(door_item.childCount(), 4)
        self.assertEqual(
            [door_item.child(index).text(0) for index in range(4)],
            ["Body right", "Body left", "Handle right", "Handle left"],
        )
        self.assertTrue(self.workspace.make_double_checkbox.isChecked())

    def test_preview_editing_controls_follow_double_asset_state(self) -> None:
        """Cursor lighting is optional and proportional editing is contextual."""

        self.assertTrue(self.workspace.cursor_light_checkbox.isChecked())
        self.assertTrue(
            self.workspace.preview_viewer.is_hover_point_light_enabled()
        )
        self.assertTrue(self.workspace.proportional_editing_checkbox.isChecked())
        self.assertTrue(self.workspace.proportional_editing_checkbox.isHidden())

        door = _door().with_double(True)
        self.workspace.set_data(DoorLibraryData(doors=[door]))

        self.assertFalse(self.workspace.proportional_editing_checkbox.isHidden())
        self.assertTrue(self.workspace.is_proportional_editing_enabled())
        self.workspace.cursor_light_checkbox.click()
        self.assertFalse(
            self.workspace.preview_viewer.is_hover_point_light_enabled()
        )

        self.workspace.set_data(DoorLibraryData(doors=[_door()]))
        self.assertTrue(self.workspace.proportional_editing_checkbox.isHidden())
        self.assertFalse(self.workspace.is_proportional_editing_enabled())

    def test_proportional_counterpart_follows_every_live_transform_preview(
        self,
    ) -> None:
        """Mirror the other leaf before the source gizmo is released."""

        door = _door().with_double(True)
        slot_models = build_default_door_slot_models(door)
        handle_model = build_door_body_model(
            0.08,
            0.12,
            0.04,
            shape="rectangular",
            arch_amount=0.0,
            geometry_name="generated_handle_fixture",
        )
        for slot_id in (DOOR_SLOT_KNOB, DOOR_SLOT_LEFT_KNOB):
            slot = door.get_slot(slot_id)
            assert slot is not None
            slot_models[slot.source_object_id] = handle_model
        complete_model = assemble_door_model(
            door,
            slot_models,
            include_unjoined=True,
        )
        right = door.get_slot(DOOR_SLOT_KNOB)
        left = door.get_slot(DOOR_SLOT_LEFT_KNOB)
        assert right is not None and left is not None

        self.workspace.set_data(DoorLibraryData(doors=[door]))
        self.workspace.select_door_slot(door.door_id, right.slot_id)
        self.workspace.set_preview_model(complete_model)

        self.workspace.preview_viewer.placed_object_transform_preview_changed.emit(
            right.source_object_id,
            (0.24, -0.03, 0.91),
            (12.0, 24.0, -36.0),
            (1.2, 0.8, 1.4),
        )

        left_preview = self.workspace.preview_viewer._placed_object_render_groups[
            left.source_object_id
        ].preview
        self.assertEqual(left_preview.world_position, (-0.24, -0.03, 0.91))
        self.assertEqual(left_preview.rotation_degrees, (12.0, -24.0, 36.0))
        self.assertEqual(left_preview.axis_scales, (1.2, 0.8, 1.4))
        persisted_left = self.workspace.data().get_door(door.door_id)
        assert persisted_left is not None
        self.assertEqual(persisted_left.get_slot(left.slot_id), left)

    def test_make_double_checkbox_requests_a_model_transition(self) -> None:
        door = _door()
        requests: list[tuple[str, bool]] = []
        self.workspace.make_double_changed.connect(
            lambda door_id, checked: requests.append((door_id, checked))
        )
        self.workspace.set_data(DoorLibraryData(doors=[door]))

        self.workspace.make_double_checkbox.click()

        self.assertEqual(requests, [(door.door_id, True)])

        updated = door.with_double(True)
        self.workspace.upsert_door(updated)
        self.workspace.make_double_checkbox.click()

        self.assertEqual(
            requests,
            [(door.door_id, True), (door.door_id, False)],
        )

    def test_ctrl_copy_and_paste_requests_the_selected_counterpart_slot(
        self,
    ) -> None:
        door = _door().with_double(True)
        requests: list[tuple[str, str, str]] = []
        self.workspace.slot_mesh_paste_requested.connect(
            lambda *values: requests.append(values)
        )
        self.workspace.set_data(DoorLibraryData(doors=[door]))

        self.assertTrue(
            self.workspace.select_door_slot(door.door_id, DOOR_SLOT_BODY)
        )
        self.workspace.copy_slot_shortcut.activated.emit()
        self.assertTrue(
            self.workspace.select_door_slot(
                door.door_id,
                DOOR_SLOT_LEFT_BODY,
            )
        )
        self.workspace.paste_slot_shortcut.activated.emit()

        self.assertEqual(
            requests,
            [(door.door_id, DOOR_SLOT_BODY, DOOR_SLOT_LEFT_BODY)],
        )

        self.assertTrue(
            self.workspace.select_door_slot(
                door.door_id,
                DOOR_SLOT_LEFT_KNOB,
            )
        )
        self.workspace.paste_slot_shortcut.activated.emit()
        self.assertEqual(
            requests,
            [(door.door_id, DOOR_SLOT_BODY, DOOR_SLOT_LEFT_BODY)],
        )

    def test_door_option_checkboxes_are_below_the_list_and_adjacent(self) -> None:
        make_double = self.workspace.make_double_checkbox
        make_double_sided = self.workspace.make_double_sided_checkbox
        option_layout = make_double_sided.parentWidget().layout()
        controls_layout = self.workspace.door_tree.parentWidget().layout()

        self.assertIs(
            make_double.parentWidget(),
            make_double_sided.parentWidget(),
        )
        assert option_layout is not None
        assert controls_layout is not None
        self.assertEqual(option_layout.indexOf(make_double), 0)
        self.assertEqual(option_layout.indexOf(make_double_sided), 1)
        self.assertGreater(
            controls_layout.indexOf(make_double.parentWidget()),
            controls_layout.indexOf(self.workspace.door_tree),
        )

    # ### Selection synchronization ###
    def test_slot_selection_emits_ids_and_survives_door_update(self) -> None:
        door = _door()
        self.workspace.set_data(DoorLibraryData(doors=[door]))
        selections: list[tuple[object, object]] = []
        self.workspace.selection_changed.connect(
            lambda door_id, slot_id: selections.append((door_id, slot_id))
        )

        self.assertTrue(
            self.workspace.select_door_slot(door.door_id, DOOR_SLOT_KNOB)
        )
        self.assertEqual(selections[-1], (door.door_id, DOOR_SLOT_KNOB))

        self.workspace.upsert_door(replace(door, name="Front door"))

        self.assertEqual(self.workspace.selected_door_id(), door.door_id)
        self.assertEqual(self.workspace.selected_slot_id(), DOOR_SLOT_KNOB)
        self.assertEqual(
            self.workspace.door_tree.topLevelItem(0).text(0),
            "Front door",
        )

    def test_select_body_can_be_requested_when_inserting_a_door(self) -> None:
        door = _door()

        self.workspace.upsert_door(door, select_body=True)

        self.assertEqual(self.workspace.selected_door_id(), door.door_id)
        self.assertEqual(self.workspace.selected_slot_id(), DOOR_SLOT_BODY)

    def test_side_duplication_controls_are_not_part_of_the_doors_tab(self) -> None:
        self.assertFalse(hasattr(self.workspace, "side_duplication_checkbox"))
        self.assertFalse(hasattr(self.workspace, "side_duplication_front_radio"))
        self.assertFalse(hasattr(self.workspace, "side_duplication_back_radio"))

    def test_double_click_requests_editing_only_for_a_component_slot(self) -> None:
        door = _door()
        self.workspace.set_data(DoorLibraryData(doors=[door]))
        requests: list[tuple[str, str]] = []
        self.workspace.slot_edit_requested.connect(
            lambda door_id, slot_id: requests.append((door_id, slot_id))
        )
        door_item = self.workspace.door_tree.topLevelItem(0)
        knob_item = door_item.child(1)

        self.workspace.door_tree.itemDoubleClicked.emit(door_item, 0)
        self.assertEqual(requests, [])

        self.workspace.door_tree.itemDoubleClicked.emit(knob_item, 0)

        self.assertEqual(requests, [(door.door_id, DOOR_SLOT_KNOB)])

    # ### Slot confirmation ###
    def test_confirm_is_only_enabled_for_pending_non_body_slot(self) -> None:
        door = _door()
        self.workspace.set_data(DoorLibraryData(doors=[door]))
        requests: list[tuple[str, str]] = []
        self.workspace.slot_confirmation_requested.connect(
            lambda door_id, slot_id: requests.append((door_id, slot_id))
        )

        self.workspace.select_door_slot(door.door_id, DOOR_SLOT_BODY)
        self.assertTrue(
            self.workspace.set_slot_position_pending(
                door.door_id,
                DOOR_SLOT_BODY,
            )
        )
        self.assertFalse(self.workspace.confirm_slot_position_button.isEnabled())

        self.workspace.select_door_slot(door.door_id, DOOR_SLOT_KNOB)
        self.assertFalse(self.workspace.confirm_slot_position_button.isEnabled())
        self.assertTrue(
            self.workspace.set_slot_position_pending(
                door.door_id,
                DOOR_SLOT_KNOB,
            )
        )
        self.assertTrue(self.workspace.confirm_slot_position_button.isEnabled())

        self.workspace.confirm_slot_position_button.click()

        self.assertEqual(requests, [(door.door_id, DOOR_SLOT_KNOB)])
        selected_item = self.workspace.door_tree.currentItem()
        assert selected_item is not None
        self.assertEqual(self.workspace.selected_slot_id(), DOOR_SLOT_KNOB)
        self.assertEqual(selected_item.text(0), "Handle")
        self.assertTrue(self.workspace.confirm_slot_position_button.isEnabled())
        self.assertEqual(
            self.workspace.confirm_slot_position_button.text(),
            "Confirm slot transform",
        )

    def test_replace_slot_updates_model_without_losing_selection(self) -> None:
        door = _door()
        self.workspace.set_data(DoorLibraryData(doors=[door]))
        self.workspace.select_door_slot(door.door_id, DOOR_SLOT_KNOB)
        knob = door.get_slot(DOOR_SLOT_KNOB)
        assert knob is not None

        self.assertTrue(
            self.workspace.replace_slot(
                door.door_id,
                knob.with_transform(
                    position_meters=(0.25, 0.02, 0.95),
                    joined=True,
                ),
            )
        )

        self.assertEqual(self.workspace.selected_slot_id(), DOOR_SLOT_KNOB)
        selected_slot = self.workspace.selected_slot()
        assert selected_slot is not None
        self.assertEqual(selected_slot.position_meters, (0.25, 0.02, 0.95))
        self.assertTrue(selected_slot.joined)

    # ### Model ownership ###
    def test_data_is_detached_and_removing_a_door_prunes_placements(self) -> None:
        door = _door()
        placement = DoorPlacement(
            placement_id="placement-1",
            door_id=door.door_id,
            level_index=0,
            doorway_id="doorway-1",
        )
        incoming = DoorLibraryData(doors=[door], placements=[placement])
        self.workspace.set_data(incoming)
        incoming.doors.clear()

        self.assertEqual(len(self.workspace.doors()), 1)
        detached = self.workspace.data()
        detached.doors.clear()
        self.assertEqual(len(self.workspace.doors()), 1)

        self.assertTrue(self.workspace.remove_door(door.door_id))
        self.assertEqual(self.workspace.data().doors, [])
        self.assertEqual(self.workspace.data().placements, [])

    # ### Complete door preview ###
    def test_preview_viewer_accepts_and_clears_a_complete_door_model(self) -> None:
        door = _door()
        slot_models = build_default_door_slot_models(door)
        complete_model = assemble_door_model(
            door,
            {
                door.slots[0].source_object_id: slot_models[DOOR_SLOT_BODY]
            },
            include_unjoined=True,
        )

        self.workspace.set_preview_model(complete_model)

        self.assertEqual(
            self.workspace.preview_viewer.objectName(),
            "door_complete_preview_viewer",
        )
        self.assertIs(self.workspace.preview_viewer.model, complete_model)

        self.workspace.set_preview_model(None)

        self.assertIsNone(self.workspace.preview_viewer.model)

    def test_preview_front_arrow_points_toward_local_negative_y_and_rebuilds(
        self,
    ) -> None:
        door = _door()
        first_model = _complete_door_model(door)

        self.workspace.set_preview_model(first_model)

        first_item = self.workspace._preview_front_direction_arrow_item
        self.assertIsNotNone(first_item)
        assert first_item is not None
        first_positions = np.asarray(first_item.pos, dtype=float).copy()
        first_bounds = np.asarray(first_model.mesh.bounds, dtype=float)
        self.assertAlmostEqual(
            first_positions[0, 0],
            float(np.mean(first_bounds[:, 0])),
        )
        self.assertLess(first_positions[0, 1], first_bounds[0, 1])
        first_direction = first_positions[1] - first_positions[0]
        self.assertLess(first_direction[1], 0.0)
        np.testing.assert_allclose(first_direction[[0, 2]], (0.0, 0.0))

        larger_model = _complete_door_model(
            replace(
                door,
                width_meters=1.8,
                height_meters=3.0,
            )
        )
        self.workspace.set_preview_model(larger_model)

        rebuilt_item = self.workspace._preview_front_direction_arrow_item
        self.assertIsNotNone(rebuilt_item)
        assert rebuilt_item is not None
        self.assertIsNot(rebuilt_item, first_item)
        self.assertNotIn(first_item, self.workspace.preview_viewer.view.items)
        rebuilt_positions = np.asarray(rebuilt_item.pos, dtype=float)
        self.assertFalse(np.allclose(rebuilt_positions, first_positions))
        self.assertLess(rebuilt_positions[1, 1], rebuilt_positions[0, 1])

        self.workspace.set_preview_model(None)

        self.assertIsNone(self.workspace._preview_front_direction_arrow_item)
        self.assertFalse(
            any(
                getattr(item, "_housemaker_doors_preview_role", None)
                == OPENING_PREVIEW_FRONT_ARROW_ROLE
                for item in self.workspace.preview_viewer.view.items
            )
        )

    def test_window_preview_front_arrow_uses_the_same_local_negative_y(self) -> None:
        window_model = assemble_window_model(
            _window(),
            include_unjoined=True,
        )

        self.workspace.set_preview_model(window_model)

        arrow_item = self.workspace._preview_front_direction_arrow_item
        self.assertIsNotNone(arrow_item)
        assert arrow_item is not None
        positions = np.asarray(arrow_item.pos, dtype=float)
        direction = positions[1] - positions[0]
        self.assertLess(direction[1], 0.0)
        np.testing.assert_allclose(direction[[0, 2]], (0.0, 0.0))

    def test_preview_front_arrow_is_doors_only_and_not_pickable(self) -> None:
        complete_model = _complete_door_model(_door())
        self.workspace.set_preview_model(complete_model)
        viewer = self.workspace.preview_viewer
        arrow_item = self.workspace._preview_front_direction_arrow_item
        self.assertIsNotNone(arrow_item)
        assert arrow_item is not None
        self.assertFalse(viewer.view._item_click_selection_enabled)
        self.assertNotIn(arrow_item, viewer._transform_gizmo_items)
        self.assertNotIn(arrow_item, viewer._placed_object_instance_gizmo_items)

        arrow_tip = np.asarray(arrow_item.pos[1], dtype=float)
        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=(
                arrow_tip + np.asarray((0.0, 0.0, 1.0)),
                np.asarray((0.0, 0.0, -1.0)),
            ),
        ):
            self.assertIsNone(viewer._pick_preview_placed_object_id(QPoint()))

        plain_viewer = GlbViewerWidget()
        self.addCleanup(plain_viewer.deleteLater)
        self.addCleanup(plain_viewer.close)
        plain_viewer.set_model(complete_model)
        self.assertFalse(
            any(
                getattr(item, "_housemaker_doors_preview_role", None)
                == OPENING_PREVIEW_FRONT_ARROW_ROLE
                for item in plain_viewer.view.items
            )
        )

    def test_selected_faces_switch_glass_button_action_for_a_door(self) -> None:
        door = _door()
        body = door.get_slot(DOOR_SLOT_BODY)
        assert body is not None
        complete_model = _complete_door_model(door)
        vertices, faces = _preview_face_geometry(
            complete_model,
            body.source_object_id,
        )
        area_requests: list[tuple[object, ...]] = []
        conversion_requests: list[tuple[object, ...]] = []
        self.workspace.glass_area_creation_requested.connect(
            lambda *values: area_requests.append(values)
        )
        self.workspace.face_glass_conversion_requested.connect(
            lambda *values: conversion_requests.append(values)
        )
        self.workspace.set_data(DoorLibraryData(doors=[door]))
        self.workspace.select_door_slot(door.door_id, DOOR_SLOT_BODY)
        self.workspace.set_preview_model(complete_model)

        self.assertTrue(
            self.workspace.set_preview_face_edit_geometry(
                body.source_object_id,
                vertices,
                faces,
            )
        )
        self.assertEqual(
            self.workspace.add_glass_area_button.text(),
            "Add glass area",
        )

        self.workspace.preview_viewer.set_selected_face_indices((3, 1))

        self.assertEqual(
            self.workspace.add_glass_area_button.text(),
            "Convert faces to glass",
        )
        self.workspace.add_glass_area_button.click()
        self.assertEqual(
            conversion_requests,
            [
                (
                    ASSET_KIND_DOOR,
                    door.door_id,
                    body.source_object_id,
                    (1, 3),
                )
            ],
        )
        self.assertEqual(area_requests, [])

        self.workspace.preview_viewer.clear_face_selection()

        self.assertEqual(
            self.workspace.add_glass_area_button.text(),
            "Add glass area",
        )
        self.workspace.add_glass_area_button.click()
        self.assertEqual(
            area_requests,
            [
                (
                    ASSET_KIND_DOOR,
                    door.door_id,
                    body.source_object_id,
                )
            ],
        )

    def test_delete_routes_selected_window_faces_before_preview_removal(
        self,
    ) -> None:
        window = _window()
        body = window.get_slot(DOOR_SLOT_BODY)
        assert body is not None
        fixture_door = _door(door_id=window.window_id)
        complete_model = _complete_door_model(fixture_door)
        vertices, faces = _preview_face_geometry(
            complete_model,
            body.source_object_id,
        )
        deletion_requests: list[tuple[object, ...]] = []
        removal_requests: list[str] = []
        self.workspace.face_deletion_requested.connect(
            lambda *values: deletion_requests.append(values)
        )
        self.workspace.preview_viewer.placed_object_removal_requested.connect(
            removal_requests.append
        )
        self.workspace.set_data(DoorLibraryData(windows=[window]))
        self.workspace.select_window_slot(window.window_id, DOOR_SLOT_BODY)
        self.workspace.set_preview_model(complete_model)
        self.assertTrue(
            self.workspace.set_preview_face_edit_geometry(
                body.source_object_id,
                vertices,
                faces,
            )
        )
        self.workspace.preview_viewer.set_selected_face_indices((2, 0))

        QTest.keyClick(
            self.workspace.preview_viewer.view,
            Qt.Key.Key_Delete,
        )
        _qt_application.processEvents()

        self.assertEqual(
            deletion_requests,
            [
                (
                    ASSET_KIND_WINDOW,
                    window.window_id,
                    body.source_object_id,
                    (0, 2),
                )
            ],
        )
        self.assertEqual(removal_requests, [])
        self.assertEqual(
            self.workspace.preview_viewer.get_selected_placed_object_ids(),
            (body.source_object_id,),
        )

    def test_f_routes_selected_vertices_to_the_exact_door_component(
        self,
    ) -> None:
        door = _door()
        body = door.get_slot(DOOR_SLOT_BODY)
        assert body is not None
        complete_model = _complete_door_model(door)
        vertices, faces = _preview_face_geometry(
            complete_model,
            body.source_object_id,
        )
        creation_requests: list[tuple[object, ...]] = []
        self.workspace.face_creation_requested.connect(
            lambda *values: creation_requests.append(values)
        )
        self.workspace.set_data(DoorLibraryData(doors=[door]))
        self.workspace.select_door_slot(door.door_id, DOOR_SLOT_BODY)
        self.workspace.set_preview_model(complete_model)
        self.assertTrue(
            self.workspace.set_preview_face_edit_geometry(
                body.source_object_id,
                vertices,
                faces,
            )
        )
        selected_vertices = tuple(int(index) for index in faces[0])
        self.workspace.preview_viewer.set_selected_vertex_indices(
            selected_vertices
        )

        QTest.keyClick(
            self.workspace.preview_viewer.view,
            Qt.Key.Key_F,
        )
        _qt_application.processEvents()

        self.assertEqual(
            creation_requests,
            [
                (
                    ASSET_KIND_DOOR,
                    door.door_id,
                    body.source_object_id,
                    selected_vertices,
                )
            ],
        )

    def test_plain_preview_click_still_selects_parts_during_face_editing(
        self,
    ) -> None:
        """Face gestures must not consume ordinary part-selection clicks."""

        door = _door()
        body = door.get_slot(DOOR_SLOT_BODY)
        knob = door.get_slot(DOOR_SLOT_KNOB)
        assert body is not None and knob is not None
        door = door.replace_slot(
            knob.with_transform(position_meters=(2.0, 0.0, 0.0))
        )
        complete_model = _complete_door_model(door)
        vertices, faces = _preview_face_geometry(
            complete_model,
            body.source_object_id,
        )
        viewer = self.workspace.preview_viewer
        self.workspace.set_data(DoorLibraryData(doors=[door]))
        self.workspace.select_door_slot(door.door_id, DOOR_SLOT_BODY)
        self.workspace.set_preview_model(complete_model)
        self.assertTrue(
            self.workspace.set_preview_face_edit_geometry(
                body.source_object_id,
                vertices,
                faces,
            )
        )
        self.workspace.resize(1000, 700)
        self.workspace.show()
        _qt_application.processEvents()

        with (
            patch.object(
                viewer.view,
                "build_camera_ray",
                return_value=((2.0, -5.0, 0.02), (0.0, 1.0, 0.0)),
            ),
            patch.object(viewer.view, "pixelSize", return_value=0.01),
        ):
            QTest.mouseClick(
                viewer.view,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
                QPoint(160, 120),
            )
            _qt_application.processEvents()

        self.assertEqual(self.workspace.selected_slot_id(), DOOR_SLOT_KNOB)
        self.assertEqual(
            viewer.get_selected_placed_object_ids(),
            (knob.source_object_id,),
        )

    def test_preview_owns_the_selected_generated_hardware_gizmo(self) -> None:
        door = _door()
        knob = door.get_slot(DOOR_SLOT_KNOB)
        assert knob is not None
        slot_models = build_default_door_slot_models(door)
        slot_models[knob.source_object_id] = build_door_body_model(
            0.08,
            0.12,
            0.04,
            shape="rectangular",
            arch_amount=0.0,
            geometry_name="generated_knob_fixture",
        )
        complete_model = assemble_door_model(
            door,
            slot_models,
            include_unjoined=True,
        )
        transforms: list[tuple[object, ...]] = []
        axis_scale_changes: list[tuple[object, ...]] = []
        self.workspace.slot_transform_changed.connect(
            lambda *values: transforms.append(values)
        )
        self.workspace.slot_axis_scales_changed.connect(
            lambda *values: axis_scale_changes.append(values)
        )

        self.workspace.set_data(DoorLibraryData(doors=[door]))
        self.workspace.select_door_slot(door.door_id, DOOR_SLOT_KNOB)
        self.workspace.set_preview_model(complete_model)

        self.assertEqual(
            self.workspace.preview_viewer.get_selected_placed_object_ids(),
            (knob.source_object_id,),
        )
        self.assertEqual(
            self.workspace.preview_viewer.get_placed_object_gizmo_mode(),
            "transform",
        )
        self.assertTrue(
            self.workspace.preview_viewer._toggle_placed_object_gizmo_mode()
        )
        self.assertEqual(
            self.workspace.preview_viewer.get_placed_object_gizmo_mode(),
            "scale",
        )
        self.assertFalse(
            self.workspace.preview_viewer.view._object_scale_wheel_steps_enabled
        )
        self.assertEqual(
            self.workspace.preview_viewer._placed_object_instance_gizmo_items,
            [],
        )
        self.workspace.preview_viewer.placed_object_transform_changed.emit(
            knob.source_object_id,
            (0.2, 0.0, 0.9),
            (0.0, 0.0, 15.0),
        )
        self.workspace.preview_viewer.placed_object_axis_scales_changed.emit(
            knob.source_object_id,
            (1.2, 0.8, 1.1),
        )
        self.assertEqual(
            transforms,
            [
                (
                    door.door_id,
                    DOOR_SLOT_KNOB,
                    (0.2, 0.0, 0.9),
                    (0.0, 0.0, 15.0),
                )
            ],
        )
        self.assertEqual(
            axis_scale_changes,
            [
                (
                    door.door_id,
                    DOOR_SLOT_KNOB,
                    (1.2, 0.8, 1.1),
                )
            ],
        )

    def test_selected_slot_rotation_text_tracks_live_preview_rotation(self) -> None:
        door = _door()
        handle = door.get_slot(DOOR_SLOT_KNOB)
        assert handle is not None
        door = door.replace_slot(
            replace(handle, rotation_degrees=(12.5, -20.0, 30.0))
        )
        handle = door.get_slot(DOOR_SLOT_KNOB)
        assert handle is not None
        self.workspace.set_data(DoorLibraryData(doors=[door]))
        self.workspace.select_door_slot(door.door_id, DOOR_SLOT_KNOB)

        controls_layout = self.workspace.door_tree.parentWidget().layout()
        assert controls_layout is not None
        self.assertEqual(
            controls_layout.indexOf(self.workspace.rotation_status_label),
            controls_layout.indexOf(self.workspace.door_tree) + 1,
        )

        self.assertEqual(
            self.workspace.rotation_status_label.text(),
            "Rotation: X 12.5 degrees, Y -20.0 degrees, Z 30.0 degrees",
        )

        self.workspace.preview_viewer.placed_object_rotation_preview_changed.emit(
            handle.source_object_id,
            (10.0, 20.0, 40.0),
        )

        self.assertEqual(
            self.workspace.rotation_status_label.text(),
            "Rotation: X 10.0 degrees, Y 20.0 degrees, Z 40.0 degrees",
        )

        self.workspace.preview_viewer.placed_object_rotation_preview_changed.emit(
            "another-object",
            (0.0, 0.0, 0.0),
        )
        self.assertEqual(
            self.workspace.rotation_status_label.text(),
            "Rotation: X 10.0 degrees, Y 20.0 degrees, Z 40.0 degrees",
        )

    def test_window_preview_emits_asset_aware_transform_and_scale(self) -> None:
        fixture_door = _door(door_id="window-preview")
        window = WindowDefinition(
            window_id="window-preview",
            name="New window 1",
            source_window_frame_id="window-frame-1",
            width_meters=fixture_door.width_meters,
            height_meters=fixture_door.height_meters,
            slots=fixture_door.slots,
        )
        handle = window.get_slot(DOOR_SLOT_KNOB)
        assert handle is not None
        transform_requests: list[tuple[object, ...]] = []
        scale_requests: list[tuple[object, ...]] = []
        legacy_transforms: list[tuple[object, ...]] = []
        self.workspace.asset_slot_transform_changed.connect(
            lambda *values: transform_requests.append(values)
        )
        self.workspace.asset_slot_axis_scales_changed.connect(
            lambda *values: scale_requests.append(values)
        )
        self.workspace.slot_transform_changed.connect(
            lambda *values: legacy_transforms.append(values)
        )
        self.workspace.set_data(DoorLibraryData(windows=[window]))
        self.workspace.select_window_slot(window.window_id, DOOR_SLOT_KNOB)
        self.workspace.set_preview_model(_complete_door_model(fixture_door))

        self.workspace.preview_viewer.placed_object_transform_changed.emit(
            handle.source_object_id,
            (0.1, 0.2, 0.3),
            (1.0, 2.0, 3.0),
        )
        self.workspace.preview_viewer.placed_object_axis_scales_changed.emit(
            handle.source_object_id,
            (1.1, 1.2, 1.3),
        )

        self.assertEqual(
            transform_requests,
            [
                (
                    ASSET_KIND_WINDOW,
                    window.window_id,
                    DOOR_SLOT_KNOB,
                    (0.1, 0.2, 0.3),
                    (1.0, 2.0, 3.0),
                )
            ],
        )
        self.assertEqual(
            scale_requests,
            [
                (
                    ASSET_KIND_WINDOW,
                    window.window_id,
                    DOOR_SLOT_KNOB,
                    (1.1, 1.2, 1.3),
                )
            ],
        )
        self.assertEqual(legacy_transforms, [])

    def test_preview_mesh_selection_selects_its_slot_before_gizmo_edits(
        self,
    ) -> None:
        door = _door()
        knob = door.get_slot(DOOR_SLOT_KNOB)
        assert knob is not None
        complete_model = _complete_door_model(door)

        self.workspace.set_data(DoorLibraryData(doors=[door]))
        self.workspace.select_door_slot(door.door_id, DOOR_SLOT_BODY)
        self.workspace.set_preview_model(complete_model)

        self.workspace.preview_viewer.placed_object_selection_changed.emit(
            knob.source_object_id
        )

        self.assertEqual(self.workspace.selected_slot_id(), DOOR_SLOT_KNOB)
        self.assertEqual(
            self.workspace.preview_viewer.get_selected_placed_object_ids(),
            (knob.source_object_id,),
        )
        self.assertEqual(
            self.workspace.preview_viewer.get_placed_object_gizmo_mode(),
            "transform",
        )
        self.assertTrue(
            self.workspace.preview_viewer._toggle_placed_object_gizmo_mode()
        )
        self.assertEqual(
            self.workspace.preview_viewer.get_placed_object_gizmo_mode(),
            "scale",
        )

    def test_clicking_hardware_twice_shows_scale_cube_handles(self) -> None:
        """Exercise the viewport click path used by the Doors tab."""

        door = _door()
        knob = door.get_slot(DOOR_SLOT_KNOB)
        assert knob is not None
        separated_knob = knob.with_transform(position_meters=(2.0, 0.0, 0.0))
        door = door.replace_slot(separated_knob)
        complete_model = _complete_door_model(door)
        viewer = self.workspace.preview_viewer

        self.workspace.set_data(DoorLibraryData(doors=[door]))
        self.workspace.set_preview_model(complete_model)
        self.workspace.resize(1000, 700)
        self.workspace.show()
        _qt_application.processEvents()

        hardware_ray = (
            (2.0, -5.0, 0.02),
            (0.0, 1.0, 0.0),
        )
        click_position = QPoint(160, 120)
        with (
            patch.object(
                viewer.view,
                "build_camera_ray",
                return_value=hardware_ray,
            ),
            patch.object(viewer.view, "pixelSize", return_value=0.01),
        ):
            QTest.mouseClick(
                viewer.view,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
                click_position,
            )
            _qt_application.processEvents()

            self.assertEqual(self.workspace.selected_slot_id(), DOOR_SLOT_KNOB)
            self.assertEqual(viewer.get_placed_object_gizmo_mode(), "transform")

            QTest.mouseClick(
                viewer.view,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
                click_position,
            )
            _qt_application.processEvents()

        self.assertEqual(viewer.get_placed_object_gizmo_mode(), "scale")
        self.assertEqual(len(viewer._transform_gizmo_items), 6)
        self.assertEqual(
            sum(
                isinstance(item, gl.GLMeshItem)
                for item in viewer._transform_gizmo_items
            ),
            3,
        )

    def test_double_clicking_a_preview_part_requests_generation_editing(
        self,
    ) -> None:
        door = _door()
        knob = door.get_slot(DOOR_SLOT_KNOB)
        assert knob is not None
        door = door.replace_slot(
            knob.with_transform(position_meters=(2.0, 0.0, 0.0))
        )
        knob = door.get_slot(DOOR_SLOT_KNOB)
        assert knob is not None
        viewer = self.workspace.preview_viewer
        requests: list[tuple[str, str]] = []
        self.workspace.slot_edit_requested.connect(
            lambda door_id, slot_id: requests.append((door_id, slot_id))
        )
        self.workspace.set_data(DoorLibraryData(doors=[door]))
        self.workspace.set_preview_model(_complete_door_model(door))
        self.workspace.resize(1000, 700)
        self.workspace.show()
        _qt_application.processEvents()

        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=((2.0, -5.0, 0.02), (0.0, 1.0, 0.0)),
        ):
            QTest.mouseDClick(
                viewer.view,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
                QPoint(160, 120),
            )
            _qt_application.processEvents()

        self.assertEqual(requests, [(door.door_id, DOOR_SLOT_KNOB)])
        self.assertEqual(self.workspace.selected_slot_id(), DOOR_SLOT_KNOB)

    def test_hardware_scale_gizmo_mode_survives_preview_reassembly(self) -> None:
        door = _door()
        knob = door.get_slot(DOOR_SLOT_KNOB)
        assert knob is not None
        complete_model = _complete_door_model(door)

        self.workspace.set_data(DoorLibraryData(doors=[door]))
        self.workspace.select_door_slot(door.door_id, DOOR_SLOT_KNOB)
        self.workspace.set_preview_model(complete_model)
        self.assertTrue(
            self.workspace.preview_viewer._toggle_placed_object_gizmo_mode()
        )

        self.workspace.set_preview_model(_complete_door_model(door))

        self.assertEqual(
            self.workspace.preview_viewer.get_selected_placed_object_ids(),
            (knob.source_object_id,),
        )
        self.assertEqual(
            self.workspace.preview_viewer.get_placed_object_gizmo_mode(),
            "scale",
        )

    def test_generated_body_uses_scale_only_gizmos(self) -> None:
        door = _door()
        body = door.get_slot(DOOR_SLOT_BODY)
        assert body is not None
        complete_model = assemble_door_model(
            door,
            {
                body.source_object_id: build_default_door_slot_models(door)[
                    DOOR_SLOT_BODY
                ]
            },
            include_unjoined=True,
        )
        axis_scale_changes: list[tuple[object, ...]] = []
        self.workspace.slot_axis_scales_changed.connect(
            lambda *values: axis_scale_changes.append(values)
        )

        self.workspace.set_data(DoorLibraryData(doors=[door]))
        self.workspace.select_door_slot(door.door_id, DOOR_SLOT_BODY)
        self.workspace.set_preview_model(complete_model)

        self.assertEqual(
            self.workspace.preview_viewer.get_selected_placed_object_ids(),
            (body.source_object_id,),
        )
        self.assertEqual(
            self.workspace.preview_viewer.get_placed_object_gizmo_mode(),
            "scale",
        )
        self.assertFalse(
            self.workspace.preview_viewer._toggle_placed_object_gizmo_mode()
        )
        self.workspace.preview_viewer.placed_object_axis_scales_changed.emit(
            body.source_object_id,
            (0.9, 1.1, 1.2),
        )

        self.assertEqual(
            axis_scale_changes,
            [(door.door_id, DOOR_SLOT_BODY, (0.9, 1.1, 1.2))],
        )

    def test_complete_door_preview_disables_mirror_fading(self) -> None:
        self.assertFalse(
            self.workspace.preview_viewer._symmetric_preview_fade_enabled
        )

    def test_wireframe_checkbox_controls_complete_door_preview(self) -> None:
        self.assertEqual(self.workspace.wireframe_checkbox.text(), "Wireframe")
        self.assertFalse(self.workspace.wireframe_checkbox.isChecked())
        self.assertFalse(
            self.workspace.preview_viewer.get_wireframe_enabled()
        )

        self.workspace.wireframe_checkbox.click()

        self.assertTrue(self.workspace.wireframe_checkbox.isChecked())
        self.assertTrue(
            self.workspace.preview_viewer.get_wireframe_enabled()
        )

        self.workspace.wireframe_checkbox.click()

        self.assertFalse(self.workspace.wireframe_checkbox.isChecked())
        self.assertFalse(
            self.workspace.preview_viewer.get_wireframe_enabled()
        )

    def test_ctrl_z_from_a_child_control_requests_doors_undo(self) -> None:
        requests: list[bool] = []
        self.workspace.undo_requested.connect(lambda: requests.append(True))
        self.workspace.show()
        self.workspace.door_tree.setFocus()
        _qt_application.processEvents()

        QTest.keyClick(
            self.workspace.door_tree,
            Qt.Key.Key_Z,
            Qt.KeyboardModifier.ControlModifier,
        )
        _qt_application.processEvents()

        self.assertEqual(requests, [True])

    def test_preview_undo_request_uses_the_same_doors_signal(self) -> None:
        requests: list[bool] = []
        self.workspace.undo_requested.connect(lambda: requests.append(True))

        self.workspace.preview_viewer.undo_requested.emit()

        self.assertEqual(requests, [True])


if __name__ == "__main__":
    unittest.main()
