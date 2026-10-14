# ### Environment setup ###
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import numpy as np
import shapely
from PIL import Image
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from shapely import Point, Polygon

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.architectural_surface_edits import insert_surface_vertex
from housemaker.blueprint_canvas import BlueprintCanvas
from housemaker.glb import convert_to_glb, convert_to_preview_model
from housemaker.main import BlueprintWorkspace
from housemaker.models import LevelData, OpenSpaceData, StairData, VertexData
from housemaker.project_io import load_project, save_project
from housemaker.surface_geometry import build_fixed_surfaces

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Event helpers ###
def _send_drag_move(canvas: BlueprintCanvas, position: QPoint) -> None:
    event = QMouseEvent(
        QEvent.Type.MouseMove,
        QPointF(position),
        QPointF(canvas.mapToGlobal(position)),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(canvas, event)


# ### Fixture helpers ###
def _build_square_level(
    index: int,
    size_pixels: float,
    *,
    image_path: Path | None = None,
) -> LevelData:
    vertex_data = VertexData()
    vertex_ids = tuple(
        vertex_data.add_vertex(*point).id
        for point in (
            (0.0, 0.0),
            (size_pixels, 0.0),
            (size_pixels, size_pixels),
            (0.0, size_pixels),
        )
    )
    for start_id, end_id in zip(
        vertex_ids,
        (*vertex_ids[1:], vertex_ids[0]),
        strict=True,
    ):
        vertex_data.add_edge(start_id, end_id)
    return LevelData(
        index=index,
        name=f"Level {index}",
        vertex_data=vertex_data,
        image_path=None if image_path is None else str(image_path),
    )


def _horizontal_surface_geometry(surface) -> object:
    return shapely.union_all(
        [
            Polygon(triangle[:, :2])
            for triangle in np.asarray(surface.mesh.triangles, dtype=float)
        ]
    )


def _scene_horizontal_area(mesh, normal_y_sign: float) -> float:
    face_mask = mesh.face_normals[:, 1] * normal_y_sign > 0.9
    return float(np.sum(mesh.area_faces[face_mask]))


def _scene_horizontal_geometry(mesh, normal_y_sign: float) -> object:
    face_mask = mesh.face_normals[:, 1] * normal_y_sign > 0.9
    return shapely.union_all(
        [
            Polygon(triangle[:, (0, 2)])
            for triangle in mesh.triangles[face_mask]
        ]
    )


def _scene_interior_vertical_area(mesh) -> float:
    vertical_mask = np.abs(mesh.face_normals[:, 1]) < 0.1
    centers = mesh.triangles_center
    minimum_x, minimum_z = mesh.bounds[0, (0, 2)]
    maximum_x, maximum_z = mesh.bounds[1, (0, 2)]
    interior_mask = np.asarray(
        [
            not any(
                np.isclose(center[coordinate_index], boundary)
                for coordinate_index, boundary in (
                    (0, minimum_x),
                    (0, maximum_x),
                    (2, minimum_z),
                    (2, maximum_z),
                )
            )
            for center in centers
        ],
        dtype=bool,
    )
    return float(np.sum(mesh.area_faces[vertical_mask & interior_mask]))


# ### Model and persistence tests ###
class OpenSpacePersistenceTests(unittest.TestCase):
    def test_open_spaces_round_trip_and_legacy_projects_default_empty(self) -> None:
        level = _build_square_level(2, 100.0)
        level.open_spaces.append(
            OpenSpaceData("opening-a", 20.0, 25.0, 70.0, 80.0)
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "open-space.housemaker"
            save_project(project_path, 2, [level])
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["levels"][0]["open_spaces"],
                [level.open_spaces[0].to_dict()],
            )

            loaded = load_project(project_path)
            self.assertEqual(loaded.levels[2].open_spaces, level.open_spaces)

            payload["levels"][0].pop("open_spaces")
            project_path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(load_project(project_path).levels[2].open_spaces, [])

    def test_loader_skips_malformed_and_duplicate_open_spaces(self) -> None:
        level = _build_square_level(2, 100.0)
        valid = OpenSpaceData("opening-a", 20.0, 20.0, 80.0, 80.0)
        level.open_spaces.append(valid)

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "open-space.housemaker"
            save_project(project_path, 2, [level])
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            payload["levels"][0]["open_spaces"].extend(
                (
                    valid.to_dict(),
                    {
                        "open_space_id": "bad-width",
                        "minimum_x": 50.0,
                        "minimum_y": 20.0,
                        "maximum_x": 50.0,
                        "maximum_y": 80.0,
                    },
                    "not-an-object",
                )
            )
            project_path.write_text(json.dumps(payload), encoding="utf-8")

            loaded = load_project(project_path)

        self.assertEqual(loaded.levels[2].open_spaces, [valid])


# ### Geometry tests ###
class OpenSpaceGeometryTests(unittest.TestCase):
    def test_owner_floor_and_immediate_lower_ceiling_use_owner_transform(
        self,
    ) -> None:
        lower = _build_square_level(1, 500.0)
        upper = _build_square_level(2, 100.0)
        upper.scale = 2.0
        upper.offset_x_meters = 5.0
        upper.open_spaces.append(
            OpenSpaceData("opening-a", 25.0, 25.0, 75.0, 75.0)
        )
        baseline_ids = {
            surface.surface_id
            for surface in build_fixed_surfaces(
                (
                    _build_square_level(1, 500.0),
                    _build_square_level(2, 100.0),
                )
            )
        }

        surfaces = {
            surface.surface_id: surface
            for surface in build_fixed_surfaces((lower, upper))
        }

        self.assertEqual(set(surfaces), baseline_ids)
        self.assertAlmostEqual(surfaces["level:1/floor"].area_square_meters, 100.0)
        self.assertAlmostEqual(surfaces["level:1/ceiling"].area_square_meters, 96.0)
        self.assertAlmostEqual(surfaces["level:2/floor"].area_square_meters, 12.0)
        self.assertAlmostEqual(surfaces["level:2/ceiling"].area_square_meters, 16.0)

        lower_ceiling = _horizontal_surface_geometry(
            surfaces["level:1/ceiling"]
        )
        self.assertFalse(lower_ceiling.covers(Point(6.0, -1.0)))
        self.assertTrue(lower_ceiling.covers(Point(1.0, -1.0)))
        preview = convert_to_preview_model((lower, upper))
        self.assertAlmostEqual(
            float(preview.scene.geometry["l1_level_1_ceiling"].area),
            96.0,
        )

    def test_preview_and_export_floor_slab_keep_the_same_hole(self) -> None:
        level = _build_square_level(2, 100.0)
        level.open_spaces.append(
            OpenSpaceData("opening-a", 25.0, 25.0, 75.0, 75.0)
        )

        preview = convert_to_preview_model([level])
        exported = convert_to_glb([level])
        preview_floor = preview.scene.geometry["l2_level_2_floor"]
        exported_floor = exported.scene.geometry["l2_level_2_floor"]

        self.assertAlmostEqual(_scene_horizontal_area(preview_floor, 1.0), 3.0)
        self.assertAlmostEqual(_scene_horizontal_area(preview_floor, -1.0), 3.0)
        self.assertAlmostEqual(_scene_horizontal_area(exported_floor, 1.0), 3.0)
        self.assertAlmostEqual(_scene_horizontal_area(exported_floor, -1.0), 3.0)
        self.assertAlmostEqual(abs(float(preview_floor.volume)), 0.9)
        self.assertAlmostEqual(abs(float(exported_floor.volume)), 0.9)
        self.assertTrue(preview_floor.is_volume)
        self.assertTrue(exported_floor.is_volume)
        self.assertAlmostEqual(_scene_interior_vertical_area(preview_floor), 1.2)
        self.assertAlmostEqual(_scene_interior_vertical_area(exported_floor), 1.2)

    def test_open_space_crossing_inner_wall_keeps_floor_fascia(self) -> None:
        level = _build_square_level(2, 100.0)
        wall_start = level.vertex_data.add_vertex(50.0, 0.0)
        wall_end = level.vertex_data.add_vertex(50.0, 100.0)
        level.vertex_data.add_edge(wall_start.id, wall_end.id)
        level.open_spaces.append(
            OpenSpaceData("opening-a", 25.0, 25.0, 75.0, 75.0)
        )

        models = (
            convert_to_preview_model((level,)),
            convert_to_glb((level,)),
        )

        for model in models:
            with self.subTest(model_type=type(model).__name__):
                floor = model.scene.geometry["l2_level_2_floor"]
                cap = _scene_horizontal_geometry(floor, 1.0)
                self.assertTrue(floor.is_watertight)
                self.assertTrue(floor.is_volume)
                self.assertTrue(cap.covers(Point(1.0, 1.0)))
                self.assertFalse(cap.covers(Point(1.25, 1.0)))

                face_centers = floor.triangles_center
                protected_fascia_mask = (
                    (np.abs(floor.face_normals[:, 1]) < 0.1)
                    & (np.abs(face_centers[:, 0] - 1.0) < 0.001)
                    & (np.abs(face_centers[:, 2] - 1.0) < 0.45)
                )
                self.assertAlmostEqual(
                    float(np.sum(floor.area_faces[protected_fascia_mask])),
                    0.6,
                )

        lower_level = _build_square_level(1, 100.0)
        surfaces = {
            surface.surface_id: surface
            for surface in build_fixed_surfaces((lower_level, level))
        }
        for surface_id in ("level:2/floor", "level:1/ceiling"):
            with self.subTest(surface_id=surface_id):
                surface_geometry = _horizontal_surface_geometry(
                    surfaces[surface_id]
                )
                self.assertTrue(surface_geometry.covers(Point(1.0, -1.0)))
                self.assertFalse(
                    surface_geometry.covers(Point(1.25, -1.0))
                )

    def test_open_space_touching_boundary_keeps_exterior_floor_fascia(
        self,
    ) -> None:
        level = _build_square_level(2, 100.0)
        level.open_spaces.append(
            OpenSpaceData("opening-a", 0.0, 25.0, 50.0, 75.0)
        )

        floor = convert_to_glb((level,)).scene.geometry[
            "l2_level_2_floor"
        ]
        cap = _scene_horizontal_geometry(floor, 1.0)
        vertical_centers = floor.triangles_center[
            np.abs(floor.face_normals[:, 1]) < 0.1
        ]

        self.assertTrue(floor.is_watertight)
        self.assertTrue(floor.is_volume)
        self.assertFalse(cap.covers(Point(0.25, 1.0)))
        self.assertTrue(
            np.any(
                np.isclose(vertical_centers[:, 0], 0.0)
                & (vertical_centers[:, 2] > 0.5)
                & (vertical_centers[:, 2] < 1.5)
            )
        )

    def test_open_space_covering_whole_footprint_removes_floor(self) -> None:
        level = _build_square_level(2, 100.0)
        level.open_spaces.append(
            OpenSpaceData("opening-a", 0.0, 0.0, 100.0, 100.0)
        )

        model = convert_to_glb((level,))
        floor_surfaces = [
            surface
            for surface in build_fixed_surfaces((level,))
            if surface.surface_type == "floor"
        ]

        self.assertNotIn("l2_level_2_floor", model.scene.geometry)
        self.assertEqual(floor_surfaces, [])

    def test_scaled_level_uses_same_fascia_guard_in_mesh_and_surfaces(
        self,
    ) -> None:
        level = _build_square_level(2, 100.0)
        level.scale = 2.0
        wall_start = level.vertex_data.add_vertex(50.0, 0.0)
        wall_end = level.vertex_data.add_vertex(50.0, 100.0)
        level.vertex_data.add_edge(wall_start.id, wall_end.id)
        level.open_spaces.append(
            OpenSpaceData("opening-a", 25.0, 25.0, 75.0, 75.0)
        )

        model = convert_to_glb((level,))
        floor_mesh = model.scene.geometry["l2_level_2_floor"].copy()
        floor_transform, _geometry_name = model.scene.graph.get(
            "l2_level_2_floor"
        )
        floor_mesh.apply_transform(floor_transform)
        fixed_floor = next(
            surface
            for surface in build_fixed_surfaces((level,))
            if surface.surface_id == "level:2/floor"
        )
        lower_level = _build_square_level(1, 100.0)
        lower_level.scale = 2.0
        lower_ceiling = next(
            surface
            for surface in build_fixed_surfaces((lower_level, level))
            if surface.surface_id == "level:1/ceiling"
        )

        self.assertAlmostEqual(
            _scene_horizontal_area(floor_mesh, 1.0),
            fixed_floor.area_square_meters,
            places=7,
        )
        self.assertAlmostEqual(fixed_floor.area_square_meters, 12.0008, places=7)
        self.assertAlmostEqual(
            lower_ceiling.area_square_meters,
            fixed_floor.area_square_meters,
            places=7,
        )

    def test_open_space_clips_existing_authored_floor(self) -> None:
        level = _build_square_level(2, 100.0)
        floor = next(
            surface
            for surface in build_fixed_surfaces([level])
            if surface.surface_id == "level:2/floor"
        )
        insertion_point = tuple(
            float(component)
            for component in np.asarray(
                floor.mesh.triangles[0],
                dtype=float,
            ).mean(axis=0)
        )
        insert_surface_vertex([level], floor.surface_id, insertion_point)
        self.assertTrue(level.editable_surfaces)
        authored_floor_ids = {
            surface.surface_id
            for surface in build_fixed_surfaces([level])
            if surface.surface_type == "floor"
        }
        level.open_spaces.append(
            OpenSpaceData("opening-a", 25.0, 25.0, 75.0, 75.0)
        )

        floor_surfaces = [
            surface
            for surface in build_fixed_surfaces([level])
            if surface.surface_type == "floor"
        ]

        self.assertAlmostEqual(
            sum(surface.area_square_meters for surface in floor_surfaces),
            3.0,
        )
        self.assertEqual(
            {surface.surface_id for surface in floor_surfaces},
            authored_floor_ids,
        )
        self.assertTrue(
            all(
                surface.source_surface_id == "level:2/floor"
                for surface in floor_surfaces
            )
        )


# ### Canvas interaction tests ###
class OpenSpaceCanvasTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.image_path = Path(self.temporary_directory.name) / "plan.png"
        Image.new("RGB", (120, 100), (30, 50, 70)).save(self.image_path)
        self.open_spaces: list[OpenSpaceData] = []
        self.canvas = BlueprintCanvas()
        self.canvas.resize(640, 480)
        self.canvas.load_blueprint(
            str(self.image_path),
            open_spaces=self.open_spaces,
        )
        self.canvas.show()
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.canvas.close()
        self.canvas.deleteLater()
        self.temporary_directory.cleanup()
        _qt_application.processEvents()

    def test_drag_select_delete_and_ctrl_z_share_persistent_history(self) -> None:
        start = self.canvas._image_to_widget(20.0, 25.0).toPoint()
        end = self.canvas._image_to_widget(90.0, 75.0).toPoint()

        self.assertTrue(self.canvas.start_open_space_placement())
        QTest.mousePress(self.canvas, Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseRelease(self.canvas, Qt.MouseButton.LeftButton, pos=end)
        _qt_application.processEvents()

        self.assertEqual(len(self.open_spaces), 1)
        self.assertFalse(self.canvas.is_open_space_placement_active())
        open_space = self.open_spaces[0]
        self.assertEqual(self.canvas.selected_open_space_id, open_space.open_space_id)
        self.assertAlmostEqual(open_space.minimum_x, 20.0, delta=0.5)
        self.assertAlmostEqual(open_space.maximum_x, 90.0, delta=0.5)

        self.canvas.selected_open_space_id = None
        center = self.canvas._image_to_widget(55.0, 50.0).toPoint()
        self.assertEqual(self.canvas._find_open_space_at(center), open_space)
        QTest.mouseClick(self.canvas, Qt.MouseButton.LeftButton, pos=center)
        self.assertEqual(self.canvas.selected_open_space_id, open_space.open_space_id)
        corner = self.canvas._image_to_widget(
            open_space.minimum_x,
            open_space.minimum_y,
        )
        self.assertIsNotNone(
            self.canvas._find_selected_open_space_corner_at(corner)
        )

        QTest.keyClick(self.canvas, Qt.Key.Key_Delete)
        self.assertEqual(self.open_spaces, [])
        QTest.keyClick(
            self.canvas,
            Qt.Key.Key_Z,
            Qt.KeyboardModifier.ControlModifier,
        )
        _qt_application.processEvents()
        self.assertEqual(self.open_spaces, [open_space])

    def test_selected_open_space_exposes_four_corner_handles(self) -> None:
        open_space = OpenSpaceData("opening-a", 20.0, 25.0, 90.0, 75.0)
        self.open_spaces.append(open_space)
        self.canvas.selected_open_space_id = open_space.open_space_id

        corners = self.canvas._get_open_space_corner_points(open_space)

        self.assertEqual(len(corners), 4)
        self.assertEqual(
            {point for _hit, point in corners},
            {
                (20.0, 25.0),
                (90.0, 25.0),
                (90.0, 75.0),
                (20.0, 75.0),
            },
        )
        for expected_hit, image_point in corners:
            actual_hit = self.canvas._find_selected_open_space_corner_at(
                self.canvas._image_to_widget(*image_point)
            )
            self.assertEqual(actual_hit, expected_hit)

    def test_selected_corner_precedes_overlapping_wall_and_stair_points(
        self,
    ) -> None:
        open_space = OpenSpaceData("opening-a", 20.0, 20.0, 80.0, 80.0)
        self.open_spaces.append(open_space)
        self.canvas.selected_open_space_id = open_space.open_space_id
        vertex = self.canvas.vertex_data.add_vertex(20.0, 20.0)
        stair = StairData(
            start_level_index=2,
            start_a_x=20.0,
            start_a_y=20.0,
            start_b_x=40.0,
            start_b_y=20.0,
            end_level_index=3,
            end_a_x=20.0,
            end_a_y=40.0,
            end_b_x=40.0,
            end_b_y=40.0,
        )
        self.canvas.set_stair_context((stair,), LevelData(index=2, name="Ground"))
        corner = self.canvas._image_to_widget(20.0, 20.0).toPoint()

        QTest.mousePress(self.canvas, Qt.MouseButton.LeftButton, pos=corner)

        hit = self.canvas._pressed_open_space_corner
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.open_space_id, open_space.open_space_id)
        self.assertIsNone(self.canvas.pressed_vertex_id)
        self.assertIsNone(self.canvas._pressed_stair_hit)
        self.assertNotEqual(vertex.id, self.canvas.selected_vertex_id)
        QTest.mouseRelease(self.canvas, Qt.MouseButton.LeftButton, pos=corner)

    def test_corner_drag_previews_and_commits_one_wall_vertex_snap(self) -> None:
        open_space = OpenSpaceData("opening-a", 20.0, 20.0, 80.0, 80.0)
        self.open_spaces.append(open_space)
        self.canvas.selected_open_space_id = open_space.open_space_id
        target_vertex = self.canvas.vertex_data.add_vertex(10.0, 12.0)
        changed: list[None] = []
        geometry_changed: list[None] = []
        self.canvas.open_spaces_changed.connect(lambda: changed.append(None))
        self.canvas.geometry_changed.connect(
            lambda: geometry_changed.append(None)
        )
        start = self.canvas._image_to_widget(20.0, 20.0).toPoint()
        target = self.canvas._image_to_widget(10.8, 12.8).toPoint()

        QTest.mousePress(self.canvas, Qt.MouseButton.LeftButton, pos=start)
        _send_drag_move(self.canvas, target)

        self.assertEqual(self.open_spaces, [open_space])
        self.assertEqual(
            self.canvas._open_space_corner_drag_preview_point,
            (target_vertex.x, target_vertex.y),
        )
        self.assertEqual(
            self.canvas._get_open_space_display_bounds(open_space),
            (10.0, 12.0, 80.0, 80.0),
        )
        self.assertEqual(changed, [])
        self.assertEqual(geometry_changed, [])

        QTest.mouseRelease(self.canvas, Qt.MouseButton.LeftButton, pos=target)

        self.assertEqual(
            self.open_spaces,
            [OpenSpaceData("opening-a", 10.0, 12.0, 80.0, 80.0)],
        )
        self.assertEqual(len(self.canvas.undo_stack), 1)
        self.assertEqual(changed, [None])
        self.assertEqual(geometry_changed, [None])
        self.assertIsNone(self.canvas._pressed_open_space_corner)
        self.assertIsNone(self.canvas._open_space_corner_drag_preview_point)

    def test_corner_drag_snaps_to_same_level_stair_point(self) -> None:
        open_space = OpenSpaceData("opening-a", 20.0, 20.0, 80.0, 80.0)
        self.open_spaces.append(open_space)
        self.canvas.selected_open_space_id = open_space.open_space_id
        level = LevelData(index=2, name="Ground")
        stair = StairData(
            start_level_index=2,
            start_a_x=10.0,
            start_a_y=12.0,
            start_b_x=40.0,
            start_b_y=12.0,
            end_level_index=3,
            end_a_x=10.0,
            end_a_y=42.0,
            end_b_x=40.0,
            end_b_y=42.0,
        )
        self.canvas.set_stair_context((stair,), level)
        start = self.canvas._image_to_widget(20.0, 20.0).toPoint()
        target = self.canvas._image_to_widget(10.8, 12.8).toPoint()

        QTest.mousePress(self.canvas, Qt.MouseButton.LeftButton, pos=start)
        _send_drag_move(self.canvas, target)
        QTest.mouseRelease(self.canvas, Qt.MouseButton.LeftButton, pos=target)

        self.assertEqual(
            self.open_spaces,
            [OpenSpaceData("opening-a", 10.0, 12.0, 80.0, 80.0)],
        )

    def test_escape_cancels_corner_drag_without_mutating_or_creating_undo(
        self,
    ) -> None:
        open_space = OpenSpaceData("opening-a", 20.0, 20.0, 80.0, 80.0)
        self.open_spaces.append(open_space)
        self.canvas.selected_open_space_id = open_space.open_space_id
        start = self.canvas._image_to_widget(20.0, 20.0).toPoint()
        target = self.canvas._image_to_widget(10.0, 12.0).toPoint()

        QTest.mousePress(self.canvas, Qt.MouseButton.LeftButton, pos=start)
        _send_drag_move(self.canvas, target)
        QTest.keyClick(self.canvas, Qt.Key.Key_Escape)

        self.assertEqual(self.open_spaces, [open_space])
        self.assertEqual(self.canvas.undo_stack, [])
        self.assertIsNone(self.canvas._pressed_open_space_corner)
        self.assertIsNone(self.canvas._open_space_corner_drag_preview_point)

    def test_rectangle_corners_snap_to_nearby_vertices(self) -> None:
        self.canvas.vertex_data.add_vertex(18.0, 22.0)
        self.canvas.vertex_data.add_vertex(92.0, 78.0)
        start = self.canvas._image_to_widget(19.5, 22.5).toPoint()
        end = self.canvas._image_to_widget(90.5, 77.5).toPoint()

        self.assertTrue(self.canvas.start_open_space_placement())
        QTest.mousePress(self.canvas, Qt.MouseButton.LeftButton, pos=start)
        snapped_start = self.canvas._open_space_drag_start_image
        self.assertIsNotNone(snapped_start)
        assert snapped_start is not None
        self.assertEqual(
            (snapped_start.x(), snapped_start.y()),
            (18.0, 22.0),
        )
        _send_drag_move(self.canvas, end)
        snapped_end = self.canvas._open_space_drag_current_image
        self.assertIsNotNone(snapped_end)
        assert snapped_end is not None
        self.assertEqual((snapped_end.x(), snapped_end.y()), (92.0, 78.0))
        self.assertEqual(len(self.canvas._open_space_drag_preview_guides), 2)
        QTest.mouseRelease(self.canvas, Qt.MouseButton.LeftButton, pos=end)
        _qt_application.processEvents()

        self.assertEqual(len(self.open_spaces), 1)
        open_space = self.open_spaces[0]
        self.assertEqual(
            (
                open_space.minimum_x,
                open_space.minimum_y,
                open_space.maximum_x,
                open_space.maximum_y,
            ),
            (18.0, 22.0, 92.0, 78.0),
        )

    def test_release_snaps_without_a_final_mouse_move(self) -> None:
        self.canvas.vertex_data.add_vertex(18.0, 22.0)
        self.canvas.vertex_data.add_vertex(92.0, 78.0)
        start = self.canvas._image_to_widget(19.5, 22.5).toPoint()
        end = self.canvas._image_to_widget(90.5, 77.5).toPoint()

        self.assertTrue(self.canvas.start_open_space_placement())
        QTest.mousePress(self.canvas, Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseRelease(self.canvas, Qt.MouseButton.LeftButton, pos=end)
        _qt_application.processEvents()

        self.assertEqual(len(self.open_spaces), 1)
        open_space = self.open_spaces[0]
        self.assertEqual(
            (
                open_space.minimum_x,
                open_space.minimum_y,
                open_space.maximum_x,
                open_space.maximum_y,
            ),
            (18.0, 22.0, 92.0, 78.0),
        )

    def test_nearby_point_wins_over_mixed_axis_candidates(self) -> None:
        first_vertex = self.canvas.vertex_data.add_vertex(20.0, 21.0)
        self.canvas.vertex_data.add_vertex(21.0, 20.0)

        preview = self.canvas._build_open_space_snap_preview(
            QPointF(20.8, 20.8)
        )

        self.assertEqual(preview.point, (20.0, 21.0))
        self.assertEqual(
            [guide.source_vertex_id for guide in preview.guides],
            [first_vertex.id, first_vertex.id],
        )

    def test_axis_guide_uses_the_nearest_aligned_vertex(self) -> None:
        self.canvas.vertex_data.add_vertex(18.0, 10.0)
        nearest_vertex = self.canvas.vertex_data.add_vertex(18.0, 90.0)

        preview = self.canvas._build_open_space_snap_preview(
            QPointF(19.5, 85.0)
        )

        self.assertEqual(preview.point, (18.0, 85.0))
        self.assertEqual(len(preview.guides), 1)
        self.assertEqual(preview.guides[0].source_vertex_id, nearest_vertex.id)

    def test_rectangle_corners_snap_to_same_level_stair_points(self) -> None:
        level = LevelData(index=2, name="Ground")
        stair = StairData(
            start_level_index=2,
            start_a_x=15.0,
            start_a_y=18.0,
            start_b_x=88.0,
            start_b_y=74.0,
            end_level_index=3,
            end_a_x=20.0,
            end_a_y=20.0,
            end_b_x=90.0,
            end_b_y=76.0,
        )
        self.canvas.set_stair_context((stair,), level)
        start = self.canvas._image_to_widget(16.5, 18.5).toPoint()
        end = self.canvas._image_to_widget(86.5, 73.5).toPoint()

        self.assertTrue(self.canvas.start_open_space_placement())
        QTest.mousePress(self.canvas, Qt.MouseButton.LeftButton, pos=start)
        _send_drag_move(self.canvas, end)
        snapped_end = self.canvas._open_space_drag_current_image
        self.assertIsNotNone(snapped_end)
        assert snapped_end is not None
        self.assertEqual((snapped_end.x(), snapped_end.y()), (88.0, 74.0))
        self.assertEqual(
            tuple(
                guide.source_point
                for guide in self.canvas._open_space_drag_preview_guides
            ),
            ((88.0, 74.0), (88.0, 74.0)),
        )
        QTest.mouseRelease(self.canvas, Qt.MouseButton.LeftButton, pos=end)
        _qt_application.processEvents()

        self.assertEqual(len(self.open_spaces), 1)
        open_space = self.open_spaces[0]
        self.assertEqual(
            (
                open_space.minimum_x,
                open_space.minimum_y,
                open_space.maximum_x,
                open_space.maximum_y,
            ),
            (15.0, 18.0, 88.0, 74.0),
        )

    def test_open_space_snap_ignores_stair_points_on_other_levels(self) -> None:
        level = LevelData(index=2, name="Ground")
        stair = StairData(
            start_level_index=3,
            start_a_x=15.0,
            start_a_y=18.0,
            start_b_x=88.0,
            start_b_y=74.0,
            end_level_index=4,
            end_a_x=20.0,
            end_a_y=20.0,
            end_b_x=90.0,
            end_b_y=76.0,
        )
        self.canvas.set_stair_context((stair,), level)
        raw_point = QPointF(16.5, 18.5)

        preview = self.canvas._build_open_space_snap_preview(raw_point)

        self.assertEqual(preview.point, (16.5, 18.5))
        self.assertEqual(preview.guides, [])

    def test_comparison_overlay_snapshots_open_spaces(self) -> None:
        comparison = _build_square_level(1, 100.0, image_path=self.image_path)
        comparison.open_spaces.append(
            OpenSpaceData("comparison-hole", 10.0, 20.0, 40.0, 60.0)
        )

        self.assertTrue(self.canvas.set_level_comparison_overlay(comparison))
        overlay = self.canvas.get_level_comparison_overlay()

        self.assertIsNotNone(overlay)
        assert overlay is not None
        self.assertEqual(overlay.open_spaces, tuple(comparison.open_spaces))


# ### Workspace history tests ###
class OpenSpaceWorkspaceTests(unittest.TestCase):
    def test_button_commit_and_workspace_undo_update_the_level(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            image_path = temporary_path / "plan.png"
            Image.new("RGB", (120, 100), (30, 50, 70)).save(image_path)
            level = _build_square_level(2, 100.0, image_path=image_path)
            workspace = BlueprintWorkspace(
                application_settings=ApplicationSettingsStore(
                    temporary_path / "settings.json"
                )
            )
            try:
                workspace._apply_project_state([level], 0)
                workspace.resize(1200, 800)
                workspace.show()
                _qt_application.processEvents()

                workspace._handle_add_open_space_clicked()
                self.assertTrue(workspace.canvas.is_open_space_placement_active())
                start = workspace.canvas._image_to_widget(20.0, 20.0).toPoint()
                end = workspace.canvas._image_to_widget(80.0, 80.0).toPoint()
                QTest.mousePress(
                    workspace.canvas,
                    Qt.MouseButton.LeftButton,
                    pos=start,
                )
                QTest.mouseRelease(
                    workspace.canvas,
                    Qt.MouseButton.LeftButton,
                    pos=end,
                )
                _qt_application.processEvents()

                self.assertEqual(len(level.open_spaces), 1)
                self.assertEqual(len(workspace._canvas_undo_stack), 1)
                self.assertEqual(
                    workspace.open_space_status_label.text(),
                    "Open spaces: 1 area",
                )

                workspace._handle_canvas_undo_requested()
                self.assertEqual(level.open_spaces, [])
                self.assertEqual(workspace._canvas_undo_stack, [])
            finally:
                workspace.shutdown()
                workspace.close()
                workspace.deleteLater()
                _qt_application.processEvents()


if __name__ == "__main__":
    unittest.main()
