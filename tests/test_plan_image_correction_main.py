# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.generation_jobs import (
    JOB_STATUS_CANCELLED,
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
)
from housemaker.main import (
    BlueprintWorkspace,
    _build_local_file_revision,
    _PlanImageCorrectionRuntime,
    _PlanImageCorrectionThread,
)
from housemaker.plan_correction_models import (
    PLAN_CORRECTION_MODEL_GPT_5_6_LUNA,
    PLAN_CORRECTION_MODEL_GPT_5_6_TERRA,
    PLAN_CORRECTION_MODEL_GPT_IMAGE_2,
    PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1,
    plan_correction_model_label,
)
from housemaker.plan_image_correction import (
    PlanCorrectionProgress,
    PlanImageCorrectionResult,
)

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Test doubles ###
class _FakeSignal:
    def __init__(self) -> None:
        self._slots: list[object] = []

    def connect(self, slot: object) -> None:
        self._slots.append(slot)

    def emit(self, value: object | None = None) -> None:
        for slot in tuple(self._slots):
            if value is None:
                slot()  # type: ignore[operator]
            else:
                slot(value)  # type: ignore[operator]


class _FakeCorrectionThread:
    """Controllable worker replacement that never invokes the correction model."""

    def __init__(
        self,
        input_path: Path,
        output_path: Path,
        api_key: str,
        model: str = PLAN_CORRECTION_MODEL_GPT_IMAGE_2,
        parent: object | None = None,
        *,
        make_walls_continuous: bool = False,
    ) -> None:
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.api_key = api_key
        self.model = model
        self.parent = parent
        del make_walls_continuous
        self.progress = _FakeSignal()
        self.finished = _FakeSignal()
        self.result: PlanImageCorrectionResult | None = None
        self.error_message: str | None = None
        self.was_cancelled = False
        self.running = False
        self.interruption_requested = False
        self.deleted_later = False

    def start(self) -> None:
        self.running = True

    def finish(self) -> None:
        self.running = False
        self.finished.emit()

    def isRunning(self) -> bool:
        return self.running

    def requestInterruption(self) -> None:
        self.interruption_requested = True

    def wait(self, _milliseconds: int) -> bool:
        self.running = False
        return True

    def deleteLater(self) -> None:
        self.deleted_later = True


# ### Main-workspace integration tests ###
class PlanImageCorrectionMainTests(unittest.TestCase):
    def setUp(self) -> None:
        self._openai_environment_patch = patch.dict(
            os.environ,
            {"OPENAI_API_KEY": ""},
        )
        self._openai_environment_patch.start()
        self.addCleanup(self._openai_environment_patch.stop)
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.temporary_path = Path(self._temporary_directory.name)
        self.workspace = BlueprintWorkspace(
            application_settings=ApplicationSettingsStore(
                self.temporary_path / "settings.json"
            )
        )
        self.workspace.resize(1200, 800)
        self.workspace.show()
        self.workspace.settings_widget.openai_api_key_edit.setText(
            "openai-plan-test-key"
        )
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        self.workspace.deleteLater()
        _qt_application.processEvents()
        self._temporary_directory.cleanup()

    def _load_source_image(
        self,
        name: str = "folded-plan.jpg",
        *,
        size: tuple[int, int] = (96, 64),
    ) -> Path:
        source_path = self.temporary_path / name
        Image.new("RGB", size, "white").save(source_path)
        self.workspace._set_current_level_image(str(source_path))
        return source_path.resolve()

    def _start_correction(self) -> tuple[_FakeCorrectionThread, str]:
        with patch(
            "housemaker.main._PlanImageCorrectionThread",
            _FakeCorrectionThread,
        ):
            self.workspace._handle_image_correction_clicked()

        runtime = self.workspace._plan_image_correction_runtimes[
            self.workspace.current_level.index
        ]
        thread = runtime.thread
        self.assertIsInstance(thread, _FakeCorrectionThread)
        return thread, runtime.job_id  # type: ignore[return-value]

    @staticmethod
    def _successful_result(
        thread: _FakeCorrectionThread,
        *,
        size: tuple[int, int] = (80, 50),
    ) -> PlanImageCorrectionResult:
        thread.output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("1", size, 1).save(thread.output_path)
        return PlanImageCorrectionResult(
            output_path=thread.output_path,
            method="openai",
            input_size=(96, 64),
            output_size=size,
            stages=("loading", "ai_correction", "complete"),
        )

    def test_plan_image_buttons_and_source_state(self) -> None:
        self.assertEqual(self.workspace.load_image_button.text(), "Load plan image")
        self.assertEqual(
            self.workspace.image_correction_button.text(),
            "Image correction",
        )
        self.assertFalse(self.workspace.image_correction_button.isEnabled())
        self.assertIn(
            "OpenAI models",
            self.workspace.image_correction_button.toolTip(),
        )
        self.assertIn(
            "Qwen Research License",
            self.workspace.image_correction_button.toolTip(),
        )

        source_path = self._load_source_image()

        self.assertTrue(self.workspace.image_correction_button.isEnabled())
        self.assertEqual(self.workspace.current_level.image_path, str(source_path))
        self.assertEqual(
            self.workspace.current_level.original_image_path,
            str(source_path),
        )

    def test_worker_forwards_the_selected_openai_model_and_key(self) -> None:
        source_path = self._load_source_image()
        output_path = self.temporary_path / "corrected.png"
        expected_result = PlanImageCorrectionResult(
            output_path=output_path,
            method="openai",
            input_size=(96, 64),
            output_size=(96, 64),
            stages=("loading", "openai_correction", "complete"),
        )
        for model in (
            PLAN_CORRECTION_MODEL_GPT_IMAGE_2,
            PLAN_CORRECTION_MODEL_GPT_5_6_LUNA,
            PLAN_CORRECTION_MODEL_GPT_5_6_TERRA,
        ):
            with self.subTest(model=model):
                worker = _PlanImageCorrectionThread(
                    source_path,
                    output_path,
                    "configured-openai-key",
                    model,
                    make_walls_continuous=True,
                )

                with patch(
                    "housemaker.main.correct_plan_image",
                    return_value=expected_result,
                ) as correct:
                    worker.run()
                
                self.assertIs(worker.result, expected_result)
                self.assertEqual(worker.model, model)
                correct.assert_called_once()
                self.assertEqual(
                    correct.call_args.args,
                    (source_path, output_path),
                )
                self.assertEqual(
                    correct.call_args.kwargs["api_key"],
                    "configured-openai-key",
                )
                self.assertEqual(correct.call_args.kwargs["model"], model)
                self.assertTrue(
                    correct.call_args.kwargs["make_walls_continuous"]
                )
                worker.deleteLater()

    def test_successful_async_correction_applies_result_and_preserves_source(
        self,
    ) -> None:
        source_path = self._load_source_image()
        thread, job_id = self._start_correction()

        self.assertEqual(thread.input_path, source_path)
        self.assertEqual(thread.api_key, "openai-plan-test-key")
        self.assertTrue(thread.running)
        self.assertFalse(self.workspace.image_correction_button.isEnabled())
        self.assertEqual(
            self.workspace.image_correction_button.text(),
            "Correcting...",
        )

        thread.progress.emit(
            PlanCorrectionProgress(
                stage="binarization",
                percent=88,
                message="Creating black-and-white plan",
            )
        )
        active_job = self.workspace.job_manager.get_job(job_id)
        self.assertIsNotNone(active_job)
        self.assertEqual(active_job.progress, 88)  # type: ignore[union-attr]
        self.assertEqual(
            active_job.stage,  # type: ignore[union-attr]
            "Creating black-and-white plan (88%)",
        )

        thread.result = self._successful_result(thread)
        thread.finish()

        level = self.workspace.current_level
        self.assertEqual(level.image_path, str(thread.output_path.resolve()))
        self.assertEqual(level.original_image_path, str(source_path))
        self.assertEqual(level.image_size_pixels, (80.0, 50.0))
        self.assertIn(
            "corrected from folded-plan.jpg", self.workspace.blueprint_name_label.text()
        )
        self.assertNotIn(level.index, self.workspace._plan_image_correction_runtimes)
        self.assertTrue(self.workspace.image_correction_button.isEnabled())
        self.assertEqual(
            self.workspace.image_correction_button.text(),
            "Image correction",
        )
        completed_job = self.workspace.job_manager.get_job(job_id)
        self.assertEqual(
            completed_job.status,  # type: ignore[union-attr]
            JOB_STATUS_COMPLETED,
        )
        self.assertEqual(
            completed_job.stage,  # type: ignore[union-attr]
            f"{plan_correction_model_label(PLAN_CORRECTION_MODEL_GPT_IMAGE_2)} "
            "plan correction completed",
        )

        second_thread, _second_job_id = self._start_correction()
        self.assertEqual(second_thread.input_path, source_path)
        self.assertNotEqual(second_thread.input_path, thread.output_path)

    def test_luna_and_terra_selection_is_forwarded_and_labeled(self) -> None:
        self._load_source_image()
        combo = self.workspace.settings_widget.plan_correction_model_combo

        for model in (
            PLAN_CORRECTION_MODEL_GPT_5_6_LUNA,
            PLAN_CORRECTION_MODEL_GPT_5_6_TERRA,
        ):
            with self.subTest(model=model):
                combo.setCurrentIndex(combo.findData(model))
                thread, job_id = self._start_correction()

                self.assertEqual(thread.model, model)
                self.assertEqual(thread.api_key, "openai-plan-test-key")

                thread.result = self._successful_result(thread)
                thread.finish()

                completed_job = self.workspace.job_manager.get_job(job_id)
                self.assertEqual(
                    completed_job.status,  # type: ignore[union-attr]
                    JOB_STATUS_COMPLETED,
                )
                self.assertEqual(
                    completed_job.stage,  # type: ignore[union-attr]
                    f"{plan_correction_model_label(model)} "
                    "plan correction completed",
                )

    def test_geometry_guard_rejects_correction_before_worker_creation(self) -> None:
        self._load_source_image()
        self.workspace.current_level.vertex_data.add_vertex(10.0, 20.0)

        with (
            patch("housemaker.main._PlanImageCorrectionThread") as thread_type,
            patch("housemaker.main.QMessageBox.warning") as warning,
        ):
            self.workspace._handle_image_correction_clicked()

        thread_type.assert_not_called()
        warning.assert_called_once()
        self.assertIn("before adding walls", warning.call_args.args[2])
        self.assertEqual(self.workspace._plan_image_correction_runtimes, {})
        self.assertEqual(self.workspace.job_manager.jobs(), ())

    def test_every_openai_model_requires_an_api_key(self) -> None:
        self._load_source_image()
        self.workspace.settings_widget.openai_api_key_edit.clear()
        output_directory = self.temporary_path / "corrected_plans"
        combo = self.workspace.settings_widget.plan_correction_model_combo

        for model in (
            PLAN_CORRECTION_MODEL_GPT_IMAGE_2,
            PLAN_CORRECTION_MODEL_GPT_5_6_LUNA,
            PLAN_CORRECTION_MODEL_GPT_5_6_TERRA,
        ):
            with (
                self.subTest(model=model),
                patch(
                    "housemaker.main._PlanImageCorrectionThread"
                ) as thread_type,
                patch("housemaker.main.QMessageBox.warning") as warning,
            ):
                combo.setCurrentIndex(combo.findData(model))
                self.workspace._handle_image_correction_clicked()

                thread_type.assert_not_called()
                warning.assert_called_once()
                self.assertEqual(
                    warning.call_args.args[1],
                    "OpenAI API key required",
                )
                self.assertIn("Settings", warning.call_args.args[2])
                self.assertEqual(
                    self.workspace._plan_image_correction_runtimes,
                    {},
                )
                self.assertEqual(self.workspace.job_manager.jobs(), ())
                self.assertFalse(output_directory.exists())

    def test_qwen_selection_does_not_require_an_openai_key(self) -> None:
        self._load_source_image()
        self.workspace.settings_widget.openai_api_key_edit.clear()
        combo = self.workspace.settings_widget.plan_correction_model_combo
        combo.setCurrentIndex(
            combo.findData(PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1)
        )

        with (
            patch(
                "housemaker.main.create_default_qwen_plan_image_editor"
            ) as create_qwen,
            patch(
                "housemaker.main._PlanImageCorrectionThread",
                _FakeCorrectionThread,
            ),
        ):
            self.workspace._handle_image_correction_clicked()

        create_qwen.assert_called_once_with()
        runtime = self.workspace._plan_image_correction_runtimes[
            self.workspace.current_level.index
        ]
        thread = runtime.thread
        self.assertEqual(thread.api_key, "")  # type: ignore[attr-defined]
        self.assertEqual(  # type: ignore[attr-defined]
            thread.model,
            PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1,
        )

    def test_failed_correction_reports_reason_and_preserves_source(self) -> None:
        source_path = self._load_source_image()
        thread, job_id = self._start_correction()
        thread.output_path.parent.mkdir(parents=True, exist_ok=True)
        thread.output_path.write_bytes(b"partial output")
        reason = "The local Qwen correction process failed."
        thread.error_message = reason

        with patch("housemaker.main.QMessageBox.critical") as critical:
            thread.finish()

        failed_job = self.workspace.job_manager.get_job(job_id)
        self.assertEqual(failed_job.status, JOB_STATUS_FAILED)  # type: ignore[union-attr]
        self.assertEqual(  # type: ignore[union-attr]
            failed_job.stage,
            f"Failed: {reason}",
        )
        self.assertEqual(self.workspace.current_level.image_path, str(source_path))
        self.assertEqual(
            self.workspace.current_level.original_image_path,
            str(source_path),
        )
        self.assertFalse(thread.output_path.exists())
        critical.assert_called_once()
        self.assertEqual(
            critical.call_args.args[1],
            "Image correction failed",
        )
        self.assertEqual(critical.call_args.args[2], reason)

    def test_changed_source_rejects_and_deletes_stale_result(self) -> None:
        source_path = self._load_source_image()
        thread, job_id = self._start_correction()
        thread.result = self._successful_result(thread)
        replacement_path = self.temporary_path / "replacement-plan.jpg"
        Image.new("RGB", (72, 48), "white").save(replacement_path)
        level = self.workspace.current_level
        level.image_path = str(replacement_path.resolve())
        level.original_image_path = str(replacement_path.resolve())

        with patch("housemaker.main.QMessageBox.warning") as warning:
            thread.finish()

        self.assertEqual(level.image_path, str(replacement_path.resolve()))
        self.assertEqual(level.original_image_path, str(replacement_path.resolve()))
        self.assertNotEqual(level.original_image_path, str(source_path))
        self.assertFalse(thread.output_path.exists())
        warning.assert_called_once()
        self.assertIn("was not applied", warning.call_args.args[2])
        failed_job = self.workspace.job_manager.get_job(job_id)
        self.assertEqual(failed_job.status, JOB_STATUS_FAILED)  # type: ignore[union-attr]

    def test_cancelled_correction_discards_output_after_worker_stops(self) -> None:
        self._load_source_image()
        thread, job_id = self._start_correction()
        thread.result = self._successful_result(thread)

        self.assertTrue(self.workspace.job_manager.cancel_job(job_id))
        self.assertTrue(thread.interruption_requested)
        self.assertEqual(
            self.workspace.image_correction_button.text(),
            "Correcting...",
        )
        thread.was_cancelled = True
        thread.finish()

        self.assertFalse(thread.output_path.exists())
        self.assertNotIn(
            self.workspace.current_level.index,
            self.workspace._plan_image_correction_runtimes,
        )
        cancelled_job = self.workspace.job_manager.get_job(job_id)
        self.assertEqual(
            cancelled_job.status,  # type: ignore[union-attr]
            JOB_STATUS_CANCELLED,
        )
        self.assertEqual(
            self.workspace.image_correction_button.text(),
            "Image correction",
        )
        self.assertTrue(self.workspace.image_correction_button.isEnabled())

    def test_cancel_and_join_discards_partial_output_on_project_replacement(
        self,
    ) -> None:
        self._load_source_image()
        thread, job_id = self._start_correction()
        thread.output_path.parent.mkdir(parents=True, exist_ok=True)
        thread.output_path.write_bytes(b"partially written correction")
        self.assertTrue(thread.output_path.exists())

        self.workspace._cancel_and_join_plan_image_corrections()

        self.assertTrue(thread.interruption_requested)
        self.assertFalse(thread.running)
        self.assertTrue(thread.deleted_later)
        self.assertFalse(thread.output_path.exists())
        self.assertEqual(self.workspace._plan_image_correction_runtimes, {})
        cancelled_job = self.workspace.job_manager.get_job(job_id)
        self.assertEqual(
            cancelled_job.status,  # type: ignore[union-attr]
            JOB_STATUS_CANCELLED,
        )
        self.assertEqual(
            self.workspace.image_correction_button.text(),
            "Image correction",
        )
        self.assertTrue(self.workspace.image_correction_button.isEnabled())

    def test_stale_finished_callback_does_not_remove_newer_runtime(self) -> None:
        source_path = self._load_source_image()
        level_index = self.workspace.current_level.index
        stale_thread = _FakeCorrectionThread(
            source_path,
            self.temporary_path / "stale.png",
            "stale-openai-key",
        )
        replacement_thread = _FakeCorrectionThread(
            source_path,
            self.temporary_path / "replacement.png",
            "replacement-openai-key",
        )
        replacement_thread.start()
        replacement_job = self.workspace.job_manager.create_job(
            kind="Image correction",
            requested_name="",
            default_name="Replacement correction",
            stage="Preparing plan image (0%)",
        )
        replacement_runtime = _PlanImageCorrectionRuntime(
            source_path=source_path,
            source_revision=_build_local_file_revision(source_path),
            output_path=replacement_thread.output_path,
            job_id=replacement_job.job_id,
            thread=replacement_thread,  # type: ignore[arg-type]
        )
        self.workspace._plan_image_correction_runtimes[level_index] = (
            replacement_runtime
        )

        self.workspace._handle_plan_image_correction_finished(
            level_index,
            "stale-job-id",
            stale_thread,  # type: ignore[arg-type]
        )

        self.assertIs(
            self.workspace._plan_image_correction_runtimes[level_index],
            replacement_runtime,
        )
        self.assertTrue(replacement_thread.running)
        self.assertEqual(
            self.workspace.image_correction_button.text(),
            "Correcting...",
        )
        self.assertFalse(self.workspace.image_correction_button.isEnabled())


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
