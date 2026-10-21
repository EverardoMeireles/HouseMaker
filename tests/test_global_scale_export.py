# ### Imports ###
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest
import trimesh

from housemaker.directional_light_state import DirectionalLightData
from housemaker.glb import (
    GeneratedModel,
    PlacedGeneratedModel,
    build_placed_generated_model_gltf_transform,
    compose_generated_model_instance_sources,
    export_glb_file,
)
from housemaker.global_scale import GlobalScale
from housemaker.scene_export import (
    GLOBAL_SCALE_METADATA_KEY,
    INSTANCE_SOURCE_SCENE_NAME,
    DoorBodyReconstruction,
    build_runtime_scene_manifest,
    write_runtime_scene_manifest,
)
from housemaker.tour_state import (
    TourCurveData,
    TourData,
    TourFloatingTooltipData,
    TourIdleCameraAnimationData,
    TourStepData,
    TourText3DData,
    TourWaitForKeyPressData,
)


# ### Fixtures and coordinate helpers ###
def _model() -> GeneratedModel:
    mesh = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
    scene = trimesh.Scene(mesh)
    return GeneratedModel(
        mesh=mesh, scene=scene, glb_bytes=scene.export(file_type="glb")
    )


def _glb_document_and_binary(payload: bytes) -> tuple[dict, bytes]:
    json_length = struct.unpack_from("<I", payload, 12)[0]
    document = json.loads(payload[20 : 20 + json_length].rstrip(b" \0"))
    return document, payload[20 + json_length :]


def _matrix(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(4, 4).T


def _tour() -> TourData:
    return TourData(
        tour_id="tour",
        name="Ground floor",
        trigger_point=(1.0, 2.0, 3.0),
        trigger_area_size=(4.0, 5.0),
        curve=TourCurveData(points=((1.0, 2.0, 3.0), (3.0, 4.0, 5.0))),
        steps=(
            TourStepData(
                step_id="step",
                progress=0.0,
                camera_target=(4.0, 5.0, 6.0),
                actions=(
                    TourText3DData(
                        component_id="text",
                        text="Welcome",
                        position=(2.0, 3.0, 4.0),
                        size_points=18.0,
                        rotation_degrees=(20.0, 35.0, 70.0),
                    ),
                    TourWaitForKeyPressData(component_id="wait"),
                    TourIdleCameraAnimationData(
                        component_id="idle",
                        pivot_point=(2.0, 4.0, 6.0),
                        radius_meters=0.1,
                    ),
                    TourFloatingTooltipData(
                        component_id="tooltip",
                        anchor_point=(3.0, 4.0, 5.0),
                    ),
                ),
            ),
        ),
    )


def _runtime_manifest(**kwargs) -> dict:
    return build_runtime_scene_manifest(
        glb_name="house.glb",
        glb_bytes=b"glb",
        source_placements={},
        instance_placements=(),
        **kwargs,
    )


# ### Exported GLB and prototype tests ###
def test_default_scale_keeps_existing_export_bytes_and_manifest(tmp_path: Path) -> None:
    path = export_glb_file(_model(), tmp_path / "house.glb")
    original_bytes = path.read_bytes()
    manifest_path = write_runtime_scene_manifest(
        path,
        source_placements={},
        instance_placements=(),
        global_scale=GlobalScale(),
    )

    assert path.read_bytes() == original_bytes
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "globalScale" not in manifest
    assert manifest["version"] == 9
    assert _runtime_manifest(global_scale=GlobalScale()) == _runtime_manifest()


def test_export_scales_rendered_scene_and_instances_but_not_prototypes(
    tmp_path: Path,
) -> None:
    source_model = _model()
    original_vertices = source_model.mesh.vertices.copy()
    instance = PlacedGeneratedModel(
        object_id="chair-instance",
        source_object_id="chair",
        object_name="Chair",
        model=source_model,
        world_position=(1.0, 2.0, 3.0),
        rotation_degrees=(20.0, 0.0, 45.0),
        axis_scales=(1.0, 0.75, 1.2),
    )
    integrated = compose_generated_model_instance_sources(_model(), (instance,))
    path = export_glb_file(integrated, tmp_path / "house.glb")
    original_bytes = path.read_bytes()
    original_document, original_binary = _glb_document_and_binary(original_bytes)
    scale = GlobalScale(2.0, 3.0, 4.0, linked=False)
    manifest_path = write_runtime_scene_manifest(
        path,
        source_placements={},
        instance_placements=(instance,),
        global_scale=scale,
    )
    first_bytes = path.read_bytes()
    document, binary = _glb_document_and_binary(first_bytes)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    wrapper = document["nodes"][document["scenes"][document["scene"]]["nodes"][0]]
    assert wrapper["scale"] == [2.0, 4.0, 3.0]
    assert wrapper["extras"][GLOBAL_SCALE_METADATA_KEY]
    library = next(
        scene
        for scene in document["scenes"]
        if scene.get("name") == INSTANCE_SOURCE_SCENE_NAME
    )
    assert set(library["nodes"]).isdisjoint(wrapper["children"])
    for index in library["nodes"]:
        assert document["nodes"][index] == original_document["nodes"][index]
    assert document["meshes"] == original_document["meshes"]
    assert document["accessors"] == original_document["accessors"]
    assert binary == original_binary
    expected = np.diag(
        (2.0, 4.0, 3.0, 1.0)
    ) @ build_placed_generated_model_gltf_transform(instance)
    np.testing.assert_allclose(
        _matrix(manifest["instanceGroups"][0]["instances"][0]["worldMatrix"]), expected
    )
    assert manifest["asset"]["sha256"] == hashlib.sha256(first_bytes).hexdigest()
    assert manifest["globalScale"]["appliedToGlb"] is True
    assert manifest["globalScale"]["appliedToRuntimeData"] is True
    assert manifest["version"] == 10
    np.testing.assert_array_equal(source_model.mesh.vertices, original_vertices)
    assert integrated.glb_bytes == original_bytes

    # Rewriting the same GLB must not multiply its transforms again.
    write_runtime_scene_manifest(
        path,
        source_placements={},
        instance_placements=(instance,),
        global_scale=scale,
    )
    assert path.read_bytes() == first_bytes
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest

    # Returning to one replaces the wrapper, restoring canonical dimensions.
    write_runtime_scene_manifest(
        path,
        source_placements={},
        instance_placements=(instance,),
        global_scale=GlobalScale(),
    )
    reset, _ = _glb_document_and_binary(path.read_bytes())
    reset_wrapper = reset["nodes"][reset["scenes"][reset["scene"]]["nodes"][0]]
    assert reset_wrapper["scale"] == [1.0, 1.0, 1.0]
    assert len(reset["nodes"]) == len(document["nodes"])


# ### Runtime spatial-feature tests ###
def test_nonuniform_export_preserves_tour_positions_and_text_affine_shape() -> None:
    tour = _tour()
    original_state = tour.to_dict()
    base = _runtime_manifest(tours=(tour,))
    scaled = _runtime_manifest(tours=(tour,), global_scale=GlobalScale(2.0, 3.0, 4.0))
    exported = scaled["tours"][0]
    assert exported["triggerPoint"] == [2.0, 12.0, -6.0]
    assert exported["triggerArea"]["center"] == exported["triggerPoint"]
    assert exported["triggerArea"]["sizeMeters"] == [8.0, 15.0]
    assert exported["curve"]["points"] == [[2.0, 12.0, -6.0], [6.0, 20.0, -12.0]]
    assert exported["steps"][0]["cameraTarget"] == [8.0, 24.0, -15.0]
    text, wait, idle, tooltip = exported["steps"][0]["actions"]
    original_text = base["tours"][0]["steps"][0]["actions"][0]
    transform = trimesh.transformations.euler_matrix(
        *np.radians(original_text["rotationDegrees"]), axes="sxyz"
    )
    transform[:3, 3] = original_text["position"]
    np.testing.assert_allclose(
        _matrix(text["worldMatrix"]), np.diag((2.0, 4.0, 3.0, 1.0)) @ transform
    )
    assert text["sizePoints"] == 18.0
    assert text["position"] == [4.0, 16.0, -9.0]
    assert wait == base["tours"][0]["steps"][0]["actions"][1]
    assert idle["pivotPoint"] == [4.0, 24.0, -12.0]
    assert idle["offsetScale"] == [2.0, 4.0, 3.0]
    assert idle["radiusMeters"] == 0.1
    assert tooltip["anchorPoint"] == [6.0, 20.0, -12.0]
    assert tour.to_dict() == original_state


def test_uniform_export_scales_text_and_idle_radius_without_new_action_semantics() -> (
    None
):
    exported = _runtime_manifest(
        tours=(_tour(),), global_scale=GlobalScale(2.0, 2.0, 2.0)
    )
    text, _wait, idle, _tooltip = exported["tours"][0]["steps"][0]["actions"]
    assert text["sizePoints"] == 36.0
    assert "worldMatrix" not in text
    assert idle["radiusMeters"] == 0.2
    assert "offsetScale" not in idle


def test_rotated_door_reflection_remains_exact_under_unequal_axes() -> None:
    reconstruction = DoorBodyReconstruction(
        placement_object_id="door",
        body_object_id="body",
        kept_side="front",
        mirror_point=(0.0, 2.0, 0.0),
        mirror_normal=(1.0, 1.0, 0.0),
        mirrored_component_object_ids=("handle",),
    )
    manifest = _runtime_manifest(
        door_body_reconstructions=(reconstruction,),
        global_scale=GlobalScale(2.0, 3.0, 4.0),
    )
    duplication = manifest["doorBodyReconstructions"][0]["sideDuplication"]
    scale = np.diag((2.0, 4.0, 3.0, 1.0))
    point = np.array((2.0, 4.0, 3.0, 1.0))
    normal = np.asarray(reconstruction.mirror_normal) / np.sqrt(2.0)
    original_reflected = point.copy()
    original_reflected[:3] -= (
        2.0 * np.dot(normal, point[:3] - reconstruction.mirror_point) * normal
    )
    np.testing.assert_allclose(
        _matrix(duplication["mirrorMatrix"]) @ scale @ point,
        scale @ original_reflected,
        atol=1e-12,
    )
    assert duplication["mirrorPlane"]["point"] == [0.0, 8.0, 0.0]
    expected_normal = normal / np.diag(scale)[:3]
    np.testing.assert_allclose(
        duplication["mirrorPlane"]["normal"],
        expected_normal / np.linalg.norm(expected_normal),
    )
    assert duplication["mirroredComponentObjectIds"] == ["handle"]
    assert reconstruction.mirror_point == (0.0, 2.0, 0.0)


def test_light_positions_and_targets_scale_while_intensity_stays_authored() -> None:
    light = DirectionalLightData(
        light_id="light",
        name="Light",
        position=(1.0, 2.0, 3.0),
        target=(4.0, 5.0, 6.0),
        intensity=2.0,
    )
    exported = _runtime_manifest(
        directional_lights=(light,),
        global_scale=GlobalScale(2.0, 3.0, 4.0),
    )["directionalLights"][0]
    assert exported["position"] == [2.0, 12.0, -6.0]
    assert exported["target"] == [8.0, 24.0, -15.0]
    assert exported["intensity"] == 2.0
    assert light.position == (1.0, 2.0, 3.0)


def test_export_rejects_an_unvalidated_global_scale() -> None:
    with pytest.raises(TypeError, match="GlobalScale"):
        _runtime_manifest(global_scale=(1.0, 2.0, 3.0))
