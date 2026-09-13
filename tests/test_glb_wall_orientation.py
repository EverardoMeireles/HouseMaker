# ### Imports ###
from __future__ import annotations

import unittest

import numpy as np

from housemaker.glb import convert_to_glb
from housemaker.models import (
    PIXEL_TO_METER,
    DoorwayData,
    LevelData,
    RoomData,
    VertexData,
)
from housemaker.surface_geometry import build_fixed_surfaces

# ### Fixture helpers ###
WallSegment = tuple[tuple[float, float], tuple[float, float]]


def _build_level_from_segments(
    segments: tuple[WallSegment, ...],
    *,
    doorways: list[DoorwayData] | None = None,
) -> LevelData:
    vertex_data = VertexData()
    vertices_by_point = {}
    for start_point, end_point in segments:
        for point in (start_point, end_point):
            if point not in vertices_by_point:
                vertices_by_point[point] = vertex_data.add_vertex(*point)
        vertex_data.add_edge(
            vertices_by_point[start_point].id,
            vertices_by_point[end_point].id,
        )
    return LevelData(
        index=2,
        name="Ground",
        vertex_data=vertex_data,
        doorways=[] if doorways is None else doorways,
    )


def _build_paired_plain_wall_level(
    *,
    doorway: bool = False,
) -> tuple[LevelData, str]:
    level = _build_level_from_segments(
        (
            ((480.0, 20.0), (20.0, 20.0)),
            ((500.0, 0.0), (0.0, 0.0)),
            ((0.0, 300.0), (500.0, 300.0)),
        ),
        doorways=(
            [
                DoorwayData(
                    center_x=250.0,
                    center_y=20.0,
                    width_meters=0.8,
                    height_meters=2.0,
                    depth_meters=0.1,
                    rotation_degrees=90.0,
                )
            ]
            if doorway
            else None
        ),
    )
    return level, "level:2/wall:1:2"


def _build_room_owned_outer_wall_level() -> tuple[LevelData, str]:
    vertex_data = VertexData()
    loops = (
        ((0.0, 0.0), (500.0, 0.0), (500.0, 500.0), (0.0, 500.0)),
        ((20.0, 20.0), (480.0, 20.0), (480.0, 480.0), (20.0, 480.0)),
    )
    loop_vertex_ids = []
    for points in loops:
        vertex_ids = tuple(vertex_data.add_vertex(*point).id for point in points)
        loop_vertex_ids.append(vertex_ids)
        for start_id, end_id in zip(
            vertex_ids,
            (*vertex_ids[1:], vertex_ids[0]),
            strict=True,
        ):
            vertex_data.add_edge(start_id, end_id)
    center = vertex_data.add_vertex(250.0, 250.0)
    outer_ids = loop_vertex_ids[0]
    room = RoomData(
        name="Outer contour room",
        vertex_ids=outer_ids,
        center_vertex_id=center.id,
        color_rgb=(120, 140, 160),
    )
    wall_id = (
        f"level:2/room:{center.id}/wall:"
        f"{min(outer_ids[0], outer_ids[1])}:"
        f"{max(outer_ids[0], outer_ids[1])}"
    )
    return (
        LevelData(
            index=2,
            name="Ground",
            vertex_data=vertex_data,
            rooms=[room],
        ),
        wall_id,
    )


def _vertical_face_normals_on_y(
    level: LevelData,
    image_y: float,
) -> np.ndarray:
    model = convert_to_glb([level])
    triangles = np.asarray(model.mesh.triangles, dtype=float)
    normals = np.asarray(model.mesh.face_normals, dtype=float)
    world_y = -image_y * PIXEL_TO_METER
    on_plane = np.all(
        np.isclose(triangles[:, :, 1], world_y, atol=1e-9),
        axis=1,
    )
    is_full_height_wall = (
        np.max(triangles[:, :, 2], axis=1)
        > float(level.floor_thickness_meters) + 1.0
    )
    return normals[
        on_plane & is_full_height_wall & (np.abs(normals[:, 1]) > 0.9)
    ]


def _fixed_surface_normals(level: LevelData, surface_id: str) -> np.ndarray:
    surface = next(
        surface
        for surface in build_fixed_surfaces([level])
        if surface.surface_id == surface_id
    )
    return np.asarray(surface.mesh.face_normals, dtype=float)


def _horizontal_face_normals_at_z(
    level: LevelData,
    z_meters: float,
) -> np.ndarray:
    model = convert_to_glb([level])
    triangles = np.asarray(model.mesh.triangles, dtype=float)
    normals = np.asarray(model.mesh.face_normals, dtype=float)
    on_plane = np.all(
        np.isclose(triangles[:, :, 2], z_meters, atol=1e-9),
        axis=1,
    )
    return normals[on_plane & (np.abs(normals[:, 2]) > 0.9)]


# ### Legacy GLB orientation tests ###
class GlbWallOrientationTests(unittest.TestCase):
    def test_plain_wall_uses_paired_orientation_and_one_winding(self) -> None:
        level, surface_id = _build_paired_plain_wall_level()

        exported_normals = _vertical_face_normals_on_y(level, 20.0)

        self.assertEqual(exported_normals.shape, (2, 3))
        np.testing.assert_allclose(
            exported_normals,
            _fixed_surface_normals(level, surface_id),
            atol=1e-9,
        )
        np.testing.assert_allclose(
            exported_normals,
            np.tile((0.0, -1.0, 0.0), (2, 1)),
            atol=1e-9,
        )

    def test_plain_wall_manual_flip_is_exported_as_an_xor(self) -> None:
        level, surface_id = _build_paired_plain_wall_level()
        level.flipped_surface_ids.add(surface_id)

        exported_normals = _vertical_face_normals_on_y(level, 20.0)

        self.assertEqual(exported_normals.shape, (2, 3))
        np.testing.assert_allclose(
            exported_normals,
            _fixed_surface_normals(level, surface_id),
            atol=1e-9,
        )
        np.testing.assert_allclose(
            exported_normals,
            np.tile((0.0, 1.0, 0.0), (2, 1)),
            atol=1e-9,
        )

    def test_room_wall_paired_orientation_and_manual_flip_match_surfaces(
        self,
    ) -> None:
        level, surface_id = _build_room_owned_outer_wall_level()
        automatic_normals = _vertical_face_normals_on_y(level, 0.0)

        self.assertEqual(automatic_normals.shape, (2, 3))
        np.testing.assert_allclose(
            automatic_normals,
            _fixed_surface_normals(level, surface_id),
            atol=1e-9,
        )
        np.testing.assert_allclose(
            automatic_normals,
            np.tile((0.0, 1.0, 0.0), (2, 1)),
            atol=1e-9,
        )

        level.flipped_surface_ids.add(surface_id)
        manual_normals = _vertical_face_normals_on_y(level, 0.0)

        self.assertEqual(manual_normals.shape, (2, 3))
        np.testing.assert_allclose(
            manual_normals,
            _fixed_surface_normals(level, surface_id),
            atol=1e-9,
        )
        np.testing.assert_allclose(
            manual_normals,
            np.tile((0.0, -1.0, 0.0), (2, 1)),
            atol=1e-9,
        )

    def test_doorway_fragments_keep_the_resolved_plain_wall_winding(self) -> None:
        level, _surface_id = _build_paired_plain_wall_level(doorway=True)

        exported_normals = _vertical_face_normals_on_y(level, 20.0)

        self.assertGreater(len(exported_normals), 2)
        np.testing.assert_allclose(
            exported_normals,
            np.tile((0.0, -1.0, 0.0), (len(exported_normals), 1)),
            atol=1e-9,
        )

    def test_raw_vertex_data_conversion_remains_double_sided(self) -> None:
        vertex_data = VertexData()
        start = vertex_data.add_vertex(0.0, 0.0)
        end = vertex_data.add_vertex(100.0, 0.0)
        vertex_data.add_edge(start.id, end.id)

        model = convert_to_glb(vertex_data)

        self.assertEqual(len(model.mesh.faces), 4)
        normal_y = np.asarray(model.mesh.face_normals, dtype=float)[:, 1]
        self.assertEqual(int(np.count_nonzero(normal_y > 0.9)), 2)
        self.assertEqual(int(np.count_nonzero(normal_y < -0.9)), 2)

    def test_manual_floor_flip_replaces_the_untextured_slab_top(self) -> None:
        level = _build_level_from_segments(
            (
                ((0.0, 0.0), (100.0, 0.0)),
                ((100.0, 0.0), (100.0, 100.0)),
                ((100.0, 100.0), (0.0, 100.0)),
                ((0.0, 100.0), (0.0, 0.0)),
            )
        )
        surface_id = "level:2/floor"
        level.flipped_surface_ids.add(surface_id)

        exported_normals = _horizontal_face_normals_at_z(
            level,
            level.floor_thickness_meters,
        )

        self.assertEqual(exported_normals.shape, (2, 3))
        np.testing.assert_allclose(
            exported_normals,
            _fixed_surface_normals(level, surface_id),
            atol=1e-9,
        )
        np.testing.assert_allclose(
            exported_normals,
            np.tile((0.0, 0.0, -1.0), (2, 1)),
            atol=1e-9,
        )

    def test_manual_ceiling_flip_is_preserved_in_untextured_export(self) -> None:
        level = _build_level_from_segments(
            (
                ((0.0, 0.0), (100.0, 0.0)),
                ((100.0, 0.0), (100.0, 100.0)),
                ((100.0, 100.0), (0.0, 100.0)),
                ((0.0, 100.0), (0.0, 0.0)),
            )
        )
        surface_id = "level:2/ceiling"
        level.flipped_surface_ids.add(surface_id)
        ceiling_surface = next(
            surface
            for surface in build_fixed_surfaces([level])
            if surface.surface_id == surface_id
        )
        fixed_normals = np.asarray(ceiling_surface.mesh.face_normals, dtype=float)
        ceiling_z = float(np.median(ceiling_surface.mesh.vertices[:, 2]))

        exported_normals = _horizontal_face_normals_at_z(level, ceiling_z)

        self.assertEqual(exported_normals.shape, fixed_normals.shape)
        np.testing.assert_allclose(exported_normals, fixed_normals, atol=1e-9)
        np.testing.assert_allclose(
            exported_normals,
            np.tile((0.0, 0.0, 1.0), (len(exported_normals), 1)),
            atol=1e-9,
        )


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
