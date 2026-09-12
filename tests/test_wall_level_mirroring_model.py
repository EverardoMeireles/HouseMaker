# ### Imports ###
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from housemaker.models import GROUND_LEVEL_INDEX, VertexData, create_default_levels
from housemaker.project_io import load_project, save_project
from housemaker.wall_mirroring import (
    WallMirrorVertexLink,
    mirror_wall_vertex_group,
)


# ### Wall-mirror project model tests ###
class WallLevelMirroringModelTests(unittest.TestCase):
    def test_loaded_vertex_allocator_advances_past_occupied_ids(self) -> None:
        vertex_data = VertexData.from_dict(
            {
                "next_vertex_id": 1,
                "vertices": [{"id": 7, "x": 10.0, "y": 20.0}],
                "edges": [],
            }
        )

        self.assertEqual(vertex_data.add_vertex(30.0, 40.0).id, 8)

    def test_vertex_data_no_longer_writes_legacy_mirror_flags(self) -> None:
        vertex_data = create_default_levels()[0].vertex_data
        vertex_data.add_vertex(10.0, 20.0)

        self.assertNotIn("wall_mirror_levels", vertex_data.to_dict())
        restored = type(vertex_data).from_dict(
            {
                **vertex_data.to_dict(),
                "wall_mirror_levels": {"1": [1, 2]},
            }
        )
        self.assertNotIn("wall_mirror_levels", restored.to_dict())

    def test_project_round_trip_preserves_links_and_materialized_topology(
        self,
    ) -> None:
        levels = create_default_levels()
        source = levels[GROUND_LEVEL_INDEX]
        first = source.vertex_data.add_vertex(10.0, 20.0)
        second = source.vertex_data.add_vertex(30.0, 20.0)
        source.vertex_data.add_edge(first.id, second.id)
        mirrored = mirror_wall_vertex_group(
            levels,
            (),
            source.index,
            (first.id, second.id),
            source.index + 1,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "mirrors.housemaker"
            save_project(
                project_path,
                GROUND_LEVEL_INDEX,
                levels,
                wall_mirror_links=mirrored.links,
            )
            restored = load_project(project_path)

        self.assertEqual(restored.wall_mirror_links, mirrored.links)
        target = restored.levels[GROUND_LEVEL_INDEX + 1]
        target_ids = {
            link.target_vertex_id for link in restored.wall_mirror_links
        }
        self.assertEqual(
            {vertex.id for vertex in target.vertex_data.vertices},
            target_ids,
        )
        self.assertTrue(target.vertex_data.has_edge(*sorted(target_ids)))

    def test_loading_legacy_flags_materializes_vertices_and_induced_edges(
        self,
    ) -> None:
        levels = create_default_levels()
        source = levels[GROUND_LEVEL_INDEX]
        first = source.vertex_data.add_vertex(10.0, 20.0)
        second = source.vertex_data.add_vertex(30.0, 20.0)
        source.vertex_data.add_edge(first.id, second.id)

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "legacy.housemaker"
            save_project(project_path, GROUND_LEVEL_INDEX, levels)
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            source_payload = next(
                level
                for level in payload["levels"]
                if level["index"] == GROUND_LEVEL_INDEX
            )
            source_payload["vertex_data"]["wall_mirror_levels"] = {
                str(first.id): [GROUND_LEVEL_INDEX + 1],
                str(second.id): [GROUND_LEVEL_INDEX + 1],
            }
            payload.pop("wall_mirror_links", None)
            project_path.write_text(json.dumps(payload), encoding="utf-8")

            restored = load_project(project_path)

        links = restored.wall_mirror_links
        self.assertEqual(len(links), 2)
        self.assertEqual(
            {
                (link.source_vertex_id, link.target_level_index)
                for link in links
            },
            {
                (first.id, GROUND_LEVEL_INDEX + 1),
                (second.id, GROUND_LEVEL_INDEX + 1),
            },
        )
        target = restored.levels[GROUND_LEVEL_INDEX + 1]
        target_ids = {link.target_vertex_id for link in links}
        self.assertTrue(target.vertex_data.has_edge(*sorted(target_ids)))

    def test_malformed_new_links_are_ignored_during_loading(self) -> None:
        levels = create_default_levels()
        source = levels[GROUND_LEVEL_INDEX]
        source_vertex = source.vertex_data.add_vertex(10.0, 20.0)
        target = levels[GROUND_LEVEL_INDEX + 1]
        target_vertex = target.vertex_data.add_vertex(10.0, 20.0)
        valid_link = WallMirrorVertexLink(
            source.index,
            source_vertex.id,
            target.index,
            target_vertex.id,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "malformed.housemaker"
            save_project(
                project_path,
                GROUND_LEVEL_INDEX,
                levels,
                wall_mirror_links=(valid_link,),
            )
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            payload["wall_mirror_links"].extend(
                (
                    {"source_level_index": True},
                    "not-an-object",
                    {
                        "source_level_index": source.index,
                        "source_vertex_id": source_vertex.id,
                        "target_level_index": target.index,
                        "target_vertex_id": 9999,
                    },
                )
            )
            project_path.write_text(json.dumps(payload), encoding="utf-8")

            restored = load_project(project_path)

        self.assertEqual(restored.wall_mirror_links, (valid_link,))


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
