# ### Imports ###
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from housemaker.camera_models import (
    CameraPose,
    InitialFirstPersonCamera,
)
from housemaker.generation_state import (
    GeneratedObjectRecord,
    GenerationData,
    MaskPoint,
    MaskStroke,
)
from housemaker.models import create_default_levels
from housemaker.project_io import load_project, save_project
from housemaker.video_source import VideoMetadata


# ### Fixture helpers ###
def _write_base_project(project_path: Path) -> dict[str, object]:
    save_project(
        project_path,
        current_level_index=2,
        levels=create_default_levels(),
    )
    return json.loads(project_path.read_text(encoding="utf-8"))


def _write_payload(project_path: Path, payload: dict[str, object]) -> None:
    project_path.write_text(json.dumps(payload), encoding="utf-8")


# ### Generation-state persistence tests ###
class GenerationStatePersistenceTests(unittest.TestCase):
    def test_generation_data_round_trips_masks_and_meshy_generated_objects(
        self,
    ) -> None:
        generation = GenerationData(
            video_metadata=VideoMetadata(
                path="walkthrough.mp4",
                frame_count=180,
                fps=29.97,
                width=1920,
                height=1080,
            ),
            current_frame_index=42,
            frame_strokes={
                42: [
                    MaskStroke(
                        mode="paint",
                        radius_normalized=0.04,
                        points=(
                            MaskPoint(0.25, 0.35),
                            MaskPoint(0.31, 0.41),
                        ),
                    ),
                    MaskStroke(
                        mode="erase",
                        radius_normalized=0.015,
                        points=(MaskPoint(0.28, 0.38),),
                    ),
                ],
                73: [
                    MaskStroke(
                        mode="paint",
                        radius_normalized=0.025,
                        points=(MaskPoint(0.62, 0.77),),
                    )
                ],
            },
            generated_objects=[
                GeneratedObjectRecord(
                    object_id="armchair-frame-91",
                    frame_index=91,
                    object_name="Meshy object",
                    provider="meshy",
                    pipeline={},
                    provider_task_id="task-meshy-123",
                    asset_path="generation_assets/task-meshy-123.glb",
                ),
            ],
        )

        self.assertEqual(
            GenerationData.from_dict(generation.to_dict()),
            generation,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "generation.json"
            save_project(
                project_path,
                current_level_index=2,
                levels=create_default_levels(),
                generation=generation,
            )
            raw_payload = json.loads(project_path.read_text(encoding="utf-8"))
            loaded_generation = load_project(project_path).generation

        self.assertEqual(raw_payload["generation"], generation.to_dict())
        self.assertEqual(loaded_generation, generation)

    def test_legacy_procedural_records_are_dropped_without_rejecting_project(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "legacy-procedural.json"
            payload = _write_base_project(project_path)
            payload["generation"] = {
                "generated_objects": [
                    {
                        "object_id": "legacy-chair",
                        "frame_index": 1,
                        "object_name": "Legacy chair",
                        "pipeline": {"schema_version": 1, "steps": []},
                        "provider": "procedural",
                    }
                ]
            }
            _write_payload(project_path, payload)

            loaded_generation = load_project(project_path).generation

        self.assertEqual(loaded_generation.generated_objects, [])


# ### Project migration tests ###
class GenerationProjectMigrationTests(unittest.TestCase):
    def test_legacy_dynamic_video_metadata_migrates_to_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "legacy-video.json"
            payload = _write_base_project(project_path)
            payload.pop("generation", None)
            payload["dynamic_generation"] = {
                "video_metadata": {
                    "source_path": "legacy-walkthrough.avi",
                    "frame_count": 240,
                    "fps": 24.0,
                    "width_pixels": 1280,
                    "height_pixels": 720,
                },
                "geometry_revision": 3,
                "alignments": [],
            }
            _write_payload(project_path, payload)

            loaded_project = load_project(project_path)

        self.assertEqual(
            loaded_project.generation.video_metadata,
            VideoMetadata(
                path="legacy-walkthrough.avi",
                frame_count=240,
                fps=24.0,
                width=1280,
                height=720,
            ),
        )
        self.assertEqual(loaded_project.generation.current_frame_index, 0)
        self.assertEqual(loaded_project.generation.frame_strokes, {})
        self.assertEqual(loaded_project.generation.generated_objects, [])

    def test_legacy_manual_frame_zero_migrates_to_initial_camera(self) -> None:
        legacy_pose = CameraPose(
            x=3.25,
            y=-1.5,
            z=1.72,
            yaw_degrees=92.0,
            pitch_degrees=-7.0,
            roll_degrees=2.0,
            fov_degrees=68.0,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "legacy-camera.json"
            payload = _write_base_project(project_path)
            payload.pop("initial_first_person_camera", None)
            payload["dynamic_generation"] = {
                "video_metadata": None,
                "alignments": [
                    {
                        "frame_index": 8,
                        "source": "manual",
                        "pose": CameraPose(x=99.0).to_dict(),
                    },
                    {
                        "frame_index": 0,
                        "source": "manual",
                        "pose": legacy_pose.to_dict(),
                    },
                ],
            }
            _write_payload(project_path, payload)

            loaded_camera = load_project(
                project_path
            ).initial_first_person_camera

        self.assertEqual(
            loaded_camera,
            InitialFirstPersonCamera(level_index=2, pose=legacy_pose),
        )

    def test_malformed_generation_payloads_fall_back_to_empty_state(self) -> None:
        malformed_payloads: tuple[object, ...] = (
            ["not", "an", "object"],
            {"current_frame_index": -1},
            {"frame_strokes": []},
            {
                "video_metadata": {
                    "path": "walkthrough.mp4",
                    "frame_count": 10,
                    "fps": 0,
                    "width": 640,
                    "height": 360,
                }
            },
            {
                "generated_objects": [
                    {
                        "object_id": "",
                        "frame_index": 0,
                        "object_name": "Chair",
                        "pipeline": {},
                    }
                ]
            },
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "malformed.json"
            base_payload = _write_base_project(project_path)
            base_payload.pop("dynamic_generation", None)

            for malformed_payload in malformed_payloads:
                with self.subTest(payload=malformed_payload):
                    payload = dict(base_payload)
                    payload["generation"] = malformed_payload
                    _write_payload(project_path, payload)

                    loaded_generation = load_project(project_path).generation

                    self.assertEqual(loaded_generation, GenerationData())


# ### Secret-boundary tests ###
class ProjectSecretBoundaryTests(unittest.TestCase):
    def test_project_json_never_contains_configured_api_keys(self) -> None:
        meshy_secret = "msy-HouseMakerNeverSerializeMeshy"
        openai_secret = "sk-HouseMakerNeverSerializeOpenAI"

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "no-secrets.json"
            with patch.dict(
                os.environ,
                {
                    "MESHY_API_KEY": meshy_secret,
                    "OPENAI_API_KEY": openai_secret,
                },
            ):
                save_project(
                    project_path,
                    current_level_index=2,
                    levels=create_default_levels(),
                    generation=GenerationData(),
                )

            serialized_project = project_path.read_text(encoding="utf-8")

        self.assertNotIn(meshy_secret, serialized_project)
        self.assertNotIn(openai_secret, serialized_project)
        self.assertNotIn("meshy_api_key", serialized_project.lower())
        self.assertNotIn("openai_api_key", serialized_project.lower())


# ### Test runner ###
if __name__ == "__main__":
    unittest.main()
