"""Room packing preserves authored pixels and separates structural ownership."""

# ### Imports ###
from __future__ import annotations

import json
import struct
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
import trimesh
from PIL import Image
from shapely import box
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.atlas_export import (
    MaterializedTextureAtlas,
    apply_texture_atlases_to_export,
    estimate_texture_atlas_draw_calls,
)
from housemaker.glb import (
    PACKED_ORM_AO_UV_ATTRIBUTE,
    Z_UP_TO_GLTF_Y_UP_TRANSFORM,
    GeneratedModel,
    PlacedGeneratedModel,
)
from housemaker.pbr_maps import ATLAS_MAP_TYPES
from housemaker.room_atlas_export import prepare_room_export_atlases
from housemaker.room_zones import RoomZone, RoomZoneLayout
from housemaker.texture_atlas_state import (
    ATLAS_PACKING_MODE_SYMMETRIC_QUARTER,
    ATLAS_PACKING_MODE_SYMMETRIC_SQUARE_PAIR,
    ATLAS_RESOLUTIONS,
    ATLAS_TYPE_OBJECT,
    ATLAS_TYPE_SURFACE,
    TextureAtlasData,
    TextureAtlasPlacement,
    TextureAtlasRecord,
    atlas_placement_pixel_regions,
)


# ### Deterministic room, geometry, and raster fixtures ###
@pytest.fixture
def rooms():
    return RoomZoneLayout(
        (
            RoomZone("room-a", 2, box(0, 0, 4, 4), 0.0, 3.0, name="L2 Room 1"),
            RoomZone("room-b", 2, box(5, 0, 9, 4), 0.0, 3.0, name="L2 Room 2"),
        )
    )


def _placement(source_id, *, x=0, y=0, size=512, half=None):
    return TextureAtlasPlacement(
        source_id,
        f"{source_id}.png",
        size,
        x,
        y,
        size,
        **(
            {
                "packing_mode": ATLAS_PACKING_MODE_SYMMETRIC_SQUARE_PAIR,
                "slot_half": half,
            }
            if half
            else {}
        ),
    )


def _atlas(
    tmp_path,
    placements,
    *,
    identifier="manual",
    resolution=2048,
    atlas_type=ATLAS_TYPE_OBJECT,
    cached_ao=False,
):
    record = TextureAtlasRecord(
        identifier,
        identifier,
        resolution,
        list(placements),
        atlas_type=atlas_type,
        **(
            {
                "surface_ao_image_path": "old-ao.png",
                "surface_ao_geometry_signature": "a" * 64,
            }
            if cached_ao
            else {}
        ),
    )
    paths = {}
    for map_index, map_type in enumerate(ATLAS_MAP_TYPES):
        image = Image.new("RGBA", (resolution, resolution), (20, 30, 40, 255))
        for index, placement in enumerate(placements):
            outer, _inner = atlas_placement_pixel_regions(placement)
            image.paste((50 + index * 30, 70 + map_index * 25, 100, 255), outer)
            image.putpixel(outer[:2], (1, 2, index + map_index + 3, 40))
        path = tmp_path / f"{identifier}.{map_type}.png"
        image.save(path)
        paths[map_type] = path
    ao_path = None
    if cached_ao:
        ao_path = tmp_path / "old-ao.png"
        Image.new("L", (resolution, resolution), 190).save(ao_path)
    return MaterializedTextureAtlas(
        record,
        paths,
        surface_ao_image_path=ao_path,
        surface_ao_geometry_signature="a" * 64 if cached_ao else None,
    )


def _object(identifier, x, y=2.0, *, source=None, width=1.0):
    mesh = trimesh.creation.box((width, 0.5, 1.0))
    model = GeneratedModel(mesh=mesh, scene=trimesh.Scene(mesh), glb_bytes=b"")
    return PlacedGeneratedModel(
        identifier,
        model,
        (x, y, 0.0),
        source_object_id=source,
    )


def _source_atlas(plan, source_id):
    return next(
        item for item in plan.atlases if item.atlas.placement_for_object(source_id)
    )


def _document(payload):
    length, chunk_type = struct.unpack_from("<II", payload, 12)
    assert chunk_type == 0x4E4F534A
    return json.loads(payload[20 : 20 + length])


# ### Source ownership, minimal resolutions, and lossless guarded pixels ###
def test_room_repack_chooses_smallest_resolutions_and_preserves_every_map_pixel(
    tmp_path, rooms
):
    original = _atlas(tmp_path, (_placement("a"), _placement("b", y=512, size=1024)))
    before = original.atlas.to_dict()
    plan = prepare_room_export_atlases(
        (original,),
        rooms,
        (_object("a", 2), _object("b", 7)),
        output_directory=tmp_path,
    )
    assert original.atlas.to_dict() == before
    assert plan.source_room_ids == {"a": ("room-a",), "b": ("room-b",)}
    assert sorted(item.atlas.resolution for item in plan.atlases) == [512, 1024]
    for source_id in ("a", "b"):
        output = _source_atlas(plan, source_id)
        old = original.atlas.placement_for_object(source_id)
        new = output.atlas.placement_for_object(source_id)
        assert new.texture_path == old.texture_path
        assert output.atlas.room_id == plan.source_room_ids[source_id][0]
        for map_type in ATLAS_MAP_TYPES:
            with (
                Image.open(original.map_paths[map_type]) as old_image,
                Image.open(output.map_paths[map_type]) as new_image,
            ):
                old_pixels = old_image.crop(atlas_placement_pixel_regions(old)[0])
                new_pixels = new_image.crop(atlas_placement_pixel_regions(new)[0])
                assert old_pixels.tobytes() == new_pixels.tobytes()


def test_reused_source_across_rooms_gets_one_shared_atlas_not_duplicate_textures(
    tmp_path, rooms
):
    original = _atlas(tmp_path, (_placement("a"),))
    plan = prepare_room_export_atlases(
        (original,),
        rooms,
        (_object("a", 2), _object("copy", 7, source="a")),
        output_directory=tmp_path,
    )
    assert plan.source_room_ids["a"] == ("room-a", "room-b")
    assert len(plan.atlases) == 1
    assert plan.atlases[0].atlas.room_id is None
    assert plan.atlases[0].atlas.room_name == "Shared objects"
    assert [p.object_id for p in plan.atlases[0].atlas.placements] == ["a"]


@pytest.mark.parametrize("x,width", ((4.5, 4.0), (20.0, 1.0)))
def test_boundary_crossing_and_unclassified_objects_are_shared(
    tmp_path, rooms, x, width
):
    original = _atlas(tmp_path, (_placement("a"),))
    plan = prepare_room_export_atlases(
        (original,),
        rooms,
        (_object("a", x, width=width),),
        output_directory=tmp_path,
    )
    assert plan.atlases[0].atlas.room_id is None


def test_unknown_copy_prevents_known_source_being_owned_by_only_one_room(
    tmp_path, rooms
):
    original = _atlas(tmp_path, (_placement("a"),))
    plan = prepare_room_export_atlases(
        (original,),
        rooms,
        (_object("a", 2), _object("copy", 30, source="a")),
        output_directory=tmp_path,
    )
    assert plan.atlases[0].atlas.room_id is None


def test_partial_half_slots_repair_into_one_small_slot_without_losing_guards(
    tmp_path, rooms
):
    placements = (_placement("a", half="left"), _placement("b", x=512, half="left"))
    original = _atlas(tmp_path, placements)
    plan = prepare_room_export_atlases(
        (original,),
        rooms,
        (_object("a", 1), _object("b", 3)),
        output_directory=tmp_path,
    )
    assert len(plan.atlases) == 1
    output = plan.atlases[0]
    assert output.atlas.resolution == 512
    assert {p.slot_half for p in output.atlas.placements} == {"left", "right"}
    for old in placements:
        new = output.atlas.placement_for_object(old.object_id)
        assert new.packing_mode == old.packing_mode
        for map_type in ATLAS_MAP_TYPES:
            with (
                Image.open(original.map_paths[map_type]) as first,
                Image.open(output.map_paths[map_type]) as second,
            ):
                assert (
                    first.crop(atlas_placement_pixel_regions(old)[0]).tobytes()
                    == second.crop(atlas_placement_pixel_regions(new)[0]).tobytes()
                )


@pytest.mark.parametrize("quarter", (False, True))
def test_orphan_library_slot_is_repacked_losslessly_instead_of_rejected(
    tmp_path,
    rooms,
    quarter,
):
    if quarter:
        placements = tuple(
            TextureAtlasPlacement(
                name,
                f"{name}.png",
                512,
                0,
                0,
                1024,
                packing_mode=ATLAS_PACKING_MODE_SYMMETRIC_QUARTER,
                slot_quadrant=quadrant,
            )
            for name, quadrant in (
                ("a", "top_left"),
                ("required-2", "top_right"),
                ("required-3", "bottom_left"),
                ("unused", "bottom_right"),
            )
        )
    else:
        placements = (_placement("a", half="left"), _placement("unused", half="right"))
    original = _atlas(tmp_path, placements)
    before = original.atlas.to_dict()
    objects = (
        (_object("a", 1), _object("required-2", 2), _object("required-3", 3))
        if quarter
        else (_object("a", 2),)
    )
    plan = prepare_room_export_atlases(
        (original,),
        rooms,
        objects,
        output_directory=tmp_path,
    )
    assert original.atlas.to_dict() == before
    assert len(plan.atlases) == 2
    library = _source_atlas(plan, "unused")
    assert library.atlas.room_id is None
    assert library.atlas.room_name == "Shared objects"
    old, new = placements[-1], library.atlas.placements[0]
    for map_type in ATLAS_MAP_TYPES:
        with (
            Image.open(original.map_paths[map_type]) as first,
            Image.open(library.map_paths[map_type]) as second,
        ):
            assert (
                first.crop(atlas_placement_pixel_regions(old)[0]).tobytes()
                == second.crop(atlas_placement_pixel_regions(new)[0]).tobytes()
            )


def test_retained_shared_library_and_new_shared_atlas_have_unique_names(
    tmp_path, rooms
):
    original = _atlas(tmp_path, (_placement("a"), _placement("unused", x=512)))
    original = replace(
        original, atlas=replace(original.atlas, name="[OBJECT] Shared objects")
    )
    plan = prepare_room_export_atlases(
        (original,),
        rooms,
        (_object("a", 20),),
        output_directory=tmp_path,
    )
    records = [item.atlas for item in plan.atlases]
    assert len({record.name.casefold() for record in records}) == 2
    TextureAtlasData(records)


# ### Cached AO, Surface preservation, and stable repeated exports ###
def test_surface_atlas_and_unused_library_placements_are_preserved(tmp_path, rooms):
    objects = _atlas(tmp_path, (_placement("a"), _placement("unused", x=512)))
    surfaces = _atlas(
        tmp_path,
        (_placement("wall"),),
        identifier="walls",
        atlas_type=ATLAS_TYPE_SURFACE,
        cached_ao=True,
    )
    plan = prepare_room_export_atlases(
        (objects, surfaces),
        rooms,
        (_object("a", 2),),
        output_directory=tmp_path,
    )
    assert surfaces in plan.atlases
    assert _source_atlas(plan, "wall") is surfaces
    library = _source_atlas(plan, "unused")
    assert library.atlas.atlas_id == objects.atlas.atlas_id
    assert library.atlas.placements == [objects.atlas.placements[1]]
    assert library.map_paths == objects.map_paths


def test_repack_invalidates_changed_object_uv1_bake_but_preserves_old_ao_file(
    tmp_path, rooms
):
    original = _atlas(tmp_path, (_placement("a"),), cached_ao=True)
    old_ao = original.surface_ao_image_path.read_bytes()
    plan = prepare_room_export_atlases(
        (original,),
        rooms,
        (_object("a", 2),),
        output_directory=tmp_path,
    )
    assert plan.warnings and "rebake" in plan.warnings[0]
    assert plan.atlases[0].surface_ao_image_path is None
    assert original.surface_ao_image_path.read_bytes() == old_ao
    assert original.atlas.surface_ao_image_path == "old-ao.png"


def test_ids_layouts_paths_and_cached_ao_stay_identical_after_adopting_room_records(
    tmp_path, rooms
):
    original = _atlas(tmp_path, (_placement("a"), _placement("b", x=512)))
    objects = (_object("a", 1), _object("b", 3))
    first = prepare_room_export_atlases(
        (original,), rooms, objects, output_directory=tmp_path
    )
    ao_path = tmp_path / "fresh-room-ao.png"
    Image.new("L", (first.atlases[0].atlas.resolution,) * 2, 180).save(ao_path)
    cached = replace(
        first.atlases[0],
        surface_ao_image_path=ao_path,
        surface_ao_geometry_signature="b" * 64,
    )
    second = prepare_room_export_atlases(
        (cached,), rooms, objects[::-1], output_directory=tmp_path
    )
    assert second.atlases[0] is cached
    assert second.atlases[0].atlas.to_dict() == first.atlases[0].atlas.to_dict()
    assert second.warnings == ()


# ### Persisted Atlas room sections and smaller manual sizes ###
@pytest.mark.parametrize("resolution", (512, 1024, 2048, 4096))
def test_room_metadata_and_supported_resolution_round_trip(resolution):
    assert resolution in ATLAS_RESOLUTIONS
    record = TextureAtlasRecord(
        "room-atlas",
        "Room",
        resolution,
        atlas_type=ATLAS_TYPE_OBJECT,
        room_id="room-a",
        room_name="L2 Room 1",
    )
    data = TextureAtlasData([record], record.atlas_id)
    restored = TextureAtlasData.from_dict(data.to_dict())
    assert restored.atlases[0].to_dict() == record.to_dict()


# ### Structural face batching by complete room ownership and material ###
def _structural_model(*, textured):
    scene = trimesh.Scene()
    triangles = (
        ((1.0, 1.0, 0.2), (2.0, 1.0, 0.2), (1.0, 2.0, 0.2)),
        ((6.0, 1.0, 0.2), (7.0, 1.0, 0.2), (6.0, 2.0, 0.2)),
        ((3.0, 1.0, 0.2), (7.0, 1.0, 0.2), (3.0, 3.0, 0.2)),
    )
    for index, vertices in enumerate(triangles):
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=((0, 1, 2),),
            process=False,
            metadata={"housemaker_surface_id": f"surface-{index}"},
        )
        if textured:
            mesh.visual = TextureVisuals(
                uv=np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
                material=PBRMaterial(name="Wall"),
            )
        mesh.apply_transform(Z_UP_TO_GLTF_Y_UP_TRANSFORM)
        scene.add_geometry(mesh, node_name=f"wall-{index}")
    return GeneratedModel(mesh=trimesh.creation.box(), scene=scene, glb_bytes=b"")


@pytest.mark.parametrize("textured", (False, True))
def test_structural_room_batches_share_material_without_splitting_triangles(
    tmp_path, rooms, textured
):
    model = _structural_model(textured=textured)
    atlases = (
        (_atlas(tmp_path, (_placement("plaster"),), atlas_type=ATLAS_TYPE_SURFACE),)
        if textured
        else ()
    )
    with patch(
        "housemaker.atlas_export.bake_placed_object_ambient_occlusion", return_value={}
    ):
        exported = apply_texture_atlases_to_export(
            model,
            atlases,
            room_layout=rooms,
            surface_source_ids={f"surface-{index}": "plaster" for index in range(3)},
        )
    assert sum(len(mesh.faces) for mesh in exported.scene.geometry.values()) == 3
    document = _document(exported.glb_bytes)
    meshes = [node for node in document["nodes"] if "mesh" in node]
    assert len(meshes) == 3
    assert {tuple(node["extras"]["housemaker_room_ids"]) for node in meshes} == {
        ("room-a",),
        ("room-b",),
        ("room-a", "room-b"),
    }
    assert len(document.get("materials", ())) == (1 if textured else 0)
    estimate = estimate_texture_atlas_draw_calls(
        model,
        tuple(item.atlas for item in atlases),
        room_layout=rooms,
        surface_source_ids={f"surface-{index}": "plaster" for index in range(3)},
    )
    assert estimate.exported_count == len(meshes)


def test_room_split_preserves_original_surface_ao_receivers_and_interpolant_uv1(
    tmp_path, rooms
):
    model = _structural_model(textured=True)
    pieces = list(model.scene.geometry.values())[:2]
    vertices = np.vstack([part.vertices for part in pieces])
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=((0, 1, 2), (3, 4, 5)),
        process=False,
        metadata={"housemaker_surface_id": "one-floor-surface"},
        visual=TextureVisuals(
            uv=np.tile(((0, 0), (1, 0), (0, 1)), (2, 1)),
            material=PBRMaterial(name="Wall"),
        ),
    )
    model = GeneratedModel(mesh=mesh, scene=trimesh.Scene(mesh), glb_bytes=b"")
    atlas = _atlas(
        tmp_path,
        (_placement("plaster"),),
        atlas_type=ATLAS_TYPE_SURFACE,
        cached_ao=True,
    )
    captured = []

    def prepared(parts_by_atlas, *_args, **_kwargs):
        captured.append(
            tuple(
                (pending.receiver_key, len(pending.fragment.faces))
                for parts in parts_by_atlas.values()
                for pending in parts
            )
        )
        fragments = {}
        for parts in parts_by_atlas.values():
            for pending in parts:
                fragment = pending.fragment.copy()
                fragment.vertex_attributes[PACKED_ORM_AO_UV_ATTRIBUTE] = (
                    np.column_stack(
                        (
                            np.arange(len(fragment.vertices)) / 5.0,
                            np.full(len(fragment.vertices), 0.5),
                        ),
                    ).astype(np.float32)
                )
                fragments[pending.receiver_key] = fragment
        return (
            fragments,
            {atlas.atlas.atlas_id: np.full((2048, 2048), 255, dtype=np.uint8)},
            {},
        )

    with patch(
        "housemaker.atlas_export._prepare_surface_ambient_occlusion_atlases",
        side_effect=prepared,
    ):
        apply_texture_atlases_to_export(
            model,
            (atlas,),
            surface_source_ids={"one-floor-surface": "plaster"},
        )
        exported = apply_texture_atlases_to_export(
            model,
            (atlas,),
            surface_source_ids={"one-floor-surface": "plaster"},
            room_layout=rooms,
        )
    assert captured[0] == captured[1]
    assert len(captured[1]) == 1
    assert len(exported.scene.geometry) == 2
    uv1 = np.vstack(
        [
            part.vertex_attributes[PACKED_ORM_AO_UV_ATTRIBUTE]
            for part in exported.scene.geometry.values()
        ]
    )
    np.testing.assert_allclose(np.sort(uv1[:, 0]), np.arange(6) / 5.0)
    document = _document(exported.glb_bytes)
    assert document["materials"][0]["occlusionTexture"]["texCoord"] == 1
    assert all(
        "TEXCOORD_1" in primitive["attributes"]
        for mesh in document["meshes"]
        for primitive in mesh["primitives"]
    )
