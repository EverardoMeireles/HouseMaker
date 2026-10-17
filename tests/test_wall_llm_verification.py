# ### Imports ###
from __future__ import annotations

import io
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

import cv2
import numpy as np

from housemaker import wall_llm_verification as verification
from housemaker.models import Edge, Vertex, VertexData
from housemaker.wall_llm_verification import (
    WallAdditionProposal,
    WallVerificationCancelled,
    WallVerificationError,
    apply_wall_proposals,
    parse_wall_verification_result,
    verify_plan_walls,
)


# ### Fixtures ###
def _existing(vertex_id: int) -> dict:
    return {"existing_vertex_id": vertex_id, "new_vertex_id": None}


def _new(vertex_id: str) -> dict:
    return {"existing_vertex_id": None, "new_vertex_id": vertex_id}


def _addition(proposal_id: str, start: dict, end: dict, **overrides: object) -> dict:
    return {
        "id": proposal_id,
        "start": start,
        "end": end,
        "confidence": 0.95,
        "reason": "A continuous wall boundary is visible on the plan.",
        **overrides,
    }


def _graph() -> VertexData:
    return VertexData(
        vertices=[Vertex(1, 10, 10), Vertex(2, 90, 10), Vertex(3, 90, 90)],
        edges=[Edge(1, 2)],
    )


def _payload() -> dict:
    return {
        "new_vertices": [{"id": "corner", "x": 10, "y": 90}],
        "additions": [
            _addition("right", _existing(2), _existing(3)),
            _addition("bottom", _existing(3), _new("corner")),
            _addition("left", _new("corner"), _existing(1)),
        ],
        "summary": "Three missing boundary edges.",
    }


class _Response(io.BytesIO):
    def __init__(self, payload: object) -> None:
        super().__init__(json.dumps(payload).encode())


def _response(payload: object) -> _Response:
    return _Response(
        {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": json.dumps(payload)}],
                }
            ],
        }
    )


# ### Add-only graph tests ###
class WallVerificationGeometryTests(unittest.TestCase):
    def parse(self, payload: object, **options: object):
        return parse_wall_verification_result(
            payload, _graph(), image_size=(100, 100), **options
        )

    def test_additions_share_new_ids_and_do_not_modify_existing_graph(self) -> None:
        graph = _graph()
        before = graph.clone()
        result = self.parse(_payload())
        self.assertEqual(len(result.proposals), 3)
        self.assertEqual(result.proposals[1].end.id, result.proposals[2].start.id)
        applied = apply_wall_proposals(graph, result.proposals)
        self.assertEqual(graph, before)
        self.assertEqual(applied.vertices[:3], before.vertices)
        self.assertEqual(applied.edges[:1], before.edges)
        self.assertEqual(len(applied.vertices), 4)
        self.assertEqual(len(applied.edges), 4)
        self.assertGreater(applied.add_vertex(5, 5).id, 4)

    def test_deselected_proposals_do_not_leave_unused_new_vertices(self) -> None:
        result = self.parse(_payload())
        applied = apply_wall_proposals(_graph(), result.proposals, ["right"])
        self.assertEqual(len(applied.vertices), 3)
        self.assertEqual(len(applied.edges), 2)
        self.assertEqual(apply_wall_proposals(_graph(), result.proposals, []), _graph())

    def test_rejects_duplicates_and_crossings_but_keeps_valid_additions(self) -> None:
        payload = _payload()
        payload["additions"].extend(
            [
                _addition("duplicate", _existing(1), _existing(2)),
                _addition("diagonal", _existing(1), _existing(3)),
                _addition("crossing", _existing(2), _new("corner")),
            ]
        )
        result = self.parse(payload)
        self.assertEqual(
            [p.proposal_id for p in result.proposals],
            ["right", "bottom", "left", "diagonal"],
        )
        self.assertEqual(len(result.rejected_reasons), 2)

    def test_rejects_t_junction_at_existing_edge_interior(self) -> None:
        payload = {
            "new_vertices": [{"id": "touch", "x": 50, "y": 10}],
            "additions": [_addition("unsafe", _existing(3), _new("touch"))],
        }
        result = self.parse(payload)
        self.assertFalse(result.proposals)
        self.assertIn("crosses", result.rejected_reasons[0])

    def test_rejects_collinear_overlap_sharing_an_endpoint(self) -> None:
        result = self.parse(
            {
                "new_vertices": [{"id": "overlap", "x": 50, "y": 10}],
                "additions": [_addition("overlap", _existing(1), _new("overlap"))],
            }
        )
        self.assertFalse(result.proposals)

    def test_nearly_identical_endpoint_reuses_existing_vertex_without_moving_it(
        self,
    ) -> None:
        result = self.parse(
            {
                "new_vertices": [{"id": "reuse", "x": 90.1, "y": 10.1}],
                "additions": [_addition("right", _new("reuse"), _existing(3))],
            }
        )
        self.assertEqual(result.proposals[0].start, Vertex(2, 90, 10))

    def test_rejects_protected_openings(self) -> None:
        result = self.parse(
            _payload(), excluded_polygons=(((80, 40), (99, 40), (99, 50), (80, 50)),)
        )
        self.assertEqual(len(result.proposals), 2)
        self.assertIn("protected opening", result.rejected_reasons[0])

    def test_rejects_invalid_opening_geometry(self) -> None:
        with self.assertRaisesRegex(WallVerificationError, "protected opening"):
            self.parse(
                _payload(), excluded_polygons=(((1, 1), (2, 2), (1, 2), (2, 1)),)
            )

    def test_rejects_nonfinite_out_of_bounds_and_boolean_coordinates(self) -> None:
        for value in (-1, 100, float("nan"), float("inf"), True):
            with self.subTest(value=value):
                result = self.parse(
                    {
                        "new_vertices": [{"id": "invalid", "x": value, "y": 30}],
                        "additions": [
                            _addition("invalid", _new("invalid"), _existing(3))
                        ],
                    }
                )
                self.assertFalse(result.proposals)
                self.assertEqual(len(result.rejected_reasons), 2)

    def test_rejects_unknown_existing_id_and_ambiguous_references(self) -> None:
        for reference in (
            _existing(100),
            _existing(True),
            {"existing_vertex_id": 1, "new_vertex_id": "corner"},
        ):
            with self.subTest(reference=reference):
                result = self.parse(
                    {
                        "new_vertices": [],
                        "additions": [_addition("invalid", reference, _existing(3))],
                    }
                )
                self.assertFalse(result.proposals)

    def test_low_confidence_and_tiny_edges_are_rejected(self) -> None:
        payload = {
            "new_vertices": [{"id": "tiny", "x": 90, "y": 11}],
            "additions": [
                _addition("uncertain", _existing(2), _existing(3), confidence=0.79),
                _addition("tiny", _existing(2), _new("tiny")),
            ],
        }
        self.assertFalse(self.parse(payload).proposals)

    def test_duplicate_temporary_ids_are_not_ambiguously_resolved(self) -> None:
        payload = _payload()
        payload["new_vertices"].append({"id": "corner", "x": 20, "y": 20})
        with self.assertRaisesRegex(WallVerificationError, "duplicate new vertex IDs"):
            self.parse(payload)

    def test_empty_review_is_valid(self) -> None:
        result = self.parse(
            {
                "new_vertices": [],
                "additions": [],
                "summary": "No confidently missing walls.",
            }
        )
        self.assertFalse(result.proposals)
        self.assertFalse(result.rejected_reasons)
        self.assertEqual(result.summary, "No confidently missing walls.")

    def test_refuses_malformed_or_excessive_response(self) -> None:
        for payload in ([], {}, {"new_vertices": [], "additions": [None] * 257}):
            with (
                self.subTest(payload_type=type(payload)),
                self.assertRaises(WallVerificationError),
            ):
                self.parse(payload)

    def test_apply_refuses_to_move_existing_vertices(self) -> None:
        proposal = WallAdditionProposal(
            "bad", Vertex(1, 20, 20), Vertex(3, 90, 90), 0.95, "bad"
        )
        graph = _graph()
        with self.assertRaisesRegex(
            WallVerificationError, "Existing wall vertices changed"
        ):
            apply_wall_proposals(graph, [proposal])
        self.assertEqual(graph, _graph())

    def test_apply_revalidates_intersections_against_current_geometry(self) -> None:
        proposals = self.parse(_payload()).proposals
        graph = _graph()
        first = graph.add_vertex(80, 50)
        second = graph.add_vertex(99, 50)
        graph.add_edge(first.id, second.id)
        with self.assertRaises(WallVerificationError):
            apply_wall_proposals(graph, proposals)


# ### Mock provider and cancellation tests ###
class WallVerificationProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.image = Path(self.directory.name) / "plan.png"
        self.assertTrue(
            cv2.imwrite(str(self.image), np.full((100, 100, 3), 255, np.uint8))
        )

    def verify(self, **kwargs: object):
        return verify_plan_walls(
            self.image, _graph(), api_key="test-secret", model="gpt-5.6-luna", **kwargs
        )

    def test_openai_request_contains_both_images_and_strict_addition_schema(
        self,
    ) -> None:
        captured = []

        def opener(request, **kwargs):
            captured.append(request)
            self.assertEqual(kwargs["timeout"], 120)
            return _response(_payload())

        result = self.verify(opener=opener)
        self.assertEqual(len(result.proposals), 3)
        body = json.loads(captured[0].data)
        content = body["input"][0]["content"]
        self.assertEqual(
            [item["type"] for item in content],
            ["input_text", "input_image", "input_image"],
        )
        self.assertNotEqual(content[1]["image_url"], content[2]["image_url"])
        self.assertIn('"original_image_width":100', content[0]["text"])
        self.assertIn("Existing vertices and edges are immutable", content[0]["text"])
        self.assertIn("intentional doorways", content[0]["text"])
        self.assertIn("untrusted data", content[0]["text"])
        self.assertIn("rather than wall centerlines", content[0]["text"])
        self.assertTrue(body["text"]["format"]["strict"])
        self.assertEqual(body["max_output_tokens"], 8192)

    def test_local_model_reuses_qwen_server_without_api_key(self) -> None:
        with patch.object(verification, "QwenCeilingHeightServer") as server_class:
            server = server_class.return_value.__enter__.return_value
            server.estimate.return_value = _payload()
            result = verify_plan_walls(
                self.image,
                _graph(),
                api_key="",
                model=verification.LOCAL_QWEN_CEILING_HEIGHT_MODEL,
            )
        self.assertEqual(len(result.proposals), 3)
        kwargs = server.estimate.call_args.kwargs
        self.assertEqual(kwargs["max_output_tokens"], 8192)
        self.assertEqual(len(kwargs["additional_images"]), 1)
        self.assertEqual(server_class.call_args.kwargs["context_tokens"], 32768)

    def test_api_credentials_and_model_are_validated_before_sending(self) -> None:
        for api_key, model in (("", "gpt-5.6-luna"), ("test", "image-generator")):
            with self.subTest(model=model), self.assertRaises(WallVerificationError):
                verify_plan_walls(self.image, _graph(), api_key=api_key, model=model)

    def test_http_error_reports_status_without_leaking_body_or_secret(self) -> None:
        def opener(*args, **kwargs):
            raise HTTPError(
                "https://example.invalid",
                400,
                "test-secret private-request-body",
                {},
                io.BytesIO(b"test-secret"),
            )

        with self.assertRaisesRegex(WallVerificationError, "HTTP 400") as caught:
            self.verify(opener=opener)
        self.assertNotIn("test-secret", str(caught.exception))
        self.assertNotIn("private-request", str(caught.exception))

    def test_http_error_includes_sanitized_provider_reason(self) -> None:
        def opener(*args, **kwargs):
            body = {
                "error": {
                    "message": (
                        "Model does not accept this image request. "
                        "Credential: test-secret, alternative sk-another-secret; "
                        "Authorization: Bearer private-token; "
                        'image "data:image/png;base64,private-payload"'
                    )
                }
            }
            raise HTTPError(
                "https://example.invalid",
                400,
                "Bad Request",
                {},
                io.BytesIO(json.dumps(body).encode()),
            )

        with self.assertRaises(WallVerificationError) as caught:
            self.verify(opener=opener)
        message = str(caught.exception)
        self.assertIn("Model does not accept this image request", message)
        for secret in (
            "test-secret",
            "sk-another-secret",
            "private-token",
            "private-payload",
        ):
            self.assertNotIn(secret, message)

    def test_http_provider_reason_is_bounded(self) -> None:
        def opener(*args, **kwargs):
            raise HTTPError(
                "https://example.invalid",
                429,
                "Rate limit",
                {},
                io.BytesIO(json.dumps({"error": {"message": "x" * 4000}}).encode()),
            )

        with self.assertRaises(WallVerificationError) as caught:
            self.verify(opener=opener)
        self.assertLess(len(str(caught.exception)), 1200)

    def test_local_request_builder_and_output_limit_are_not_capped_to_500(self) -> None:
        from housemaker.qwen_ceiling_height_estimation import (
            QwenCeilingHeightError,
            build_local_qwen_request_body,
            extract_local_qwen_result,
        )

        request = build_local_qwen_request_body(
            "data:image/png;base64,fixture",
            max_output_tokens=8192,
            prompt="Review walls.",
            result_schema=verification.WALL_VERIFICATION_RESULT_SCHEMA,
        )
        self.assertEqual(request["max_tokens"], 8192)
        with self.assertRaisesRegex(QwenCeilingHeightError, "output-token limit"):
            extract_local_qwen_result(
                {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": json.dumps(_payload())},
                        }
                    ],
                }
            )

    def test_resized_image_prompt_keeps_original_coordinate_dimensions(self) -> None:
        large = np.full((2400, 1200, 3), 255, np.uint8)
        self.assertTrue(cv2.imwrite(str(self.image), large))
        requests = []

        def opener(request, **kwargs):
            requests.append(json.loads(request.data))
            return _response(_payload())

        self.verify(opener=opener)
        prompt = requests[0]["input"][0]["content"][0]["text"]
        self.assertIn('"original_image_width":1200', prompt)
        self.assertIn('"original_image_height":2400', prompt)
        self.assertIn('"provided_image_width":1024', prompt)
        self.assertIn('"provided_image_height":2048', prompt)

    def test_no_fallback_on_model_refusal_or_incomplete_response(self) -> None:
        for payload in (
            {"status": "incomplete", "output": []},
            {
                "status": "completed",
                "output": [{"content": [{"type": "refusal", "refusal": "No"}]}],
            },
            {"status": "completed", "output": []},
            {"status": "failed", "output": []},
            {
                "status": "completed",
                "output": [{"content": [{"type": "output_text", "text": "not json"}]}],
            },
        ):
            with (
                self.subTest(payload=payload),
                self.assertRaises(WallVerificationError),
            ):
                self.verify(
                    opener=lambda *args, response_payload=payload, **kwargs: _Response(
                        response_payload
                    )
                )

    def test_pre_cancelled_request_does_not_read_image_or_call_provider(self) -> None:
        opener = MagicMock()
        with self.assertRaises(WallVerificationCancelled):
            self.verify(cancellation_check=lambda: True, opener=opener)
        opener.assert_not_called()

    def test_blocked_network_request_can_be_cancelled_promptly(self) -> None:
        cancelled = threading.Event()
        started = threading.Event()
        release = threading.Event()

        def opener(*args, **kwargs):
            started.set()
            release.wait(3)
            return _response(_payload())

        timer = threading.Timer(0.15, cancelled.set)
        timer.start()
        before = time.monotonic()
        try:
            with self.assertRaises(WallVerificationCancelled):
                self.verify(cancellation_check=cancelled.is_set, opener=opener)
            self.assertTrue(started.is_set())
            self.assertLess(time.monotonic() - before, 1)
        finally:
            release.set()
            timer.cancel()

    def test_missing_or_invalid_image_reports_a_specific_error(self) -> None:
        self.image.write_bytes(b"not an image")
        with self.assertRaisesRegex(WallVerificationError, "decoded"):
            self.verify()
        self.image.unlink()
        with self.assertRaisesRegex(WallVerificationError, "read"):
            self.verify()


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
