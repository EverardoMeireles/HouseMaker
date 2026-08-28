# ### Environment setup ###
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PySide6.QtWidgets import QApplication, QWidget

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.generation_jobs import (
    JOB_STATUS_CANCELLED,
    JOB_STATUS_COMPLETED,
    JOB_STATUS_RUNNING,
    GenerationJobManager,
    JobsWindow,
)
from housemaker.main import BlueprintWorkspace
from housemaker.settings_widget import (
    JOBS_WINDOW_SCREEN_SETTING_KEY,
    Fullscreen3DViewerScreenOption,
    GenerationServiceSettings,
    SettingsWidget,
    connected_fullscreen_3d_viewer_display_options,
    read_jobs_window_screen_id,
    resolve_fullscreen_3d_viewer_screen,
)


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Job manager tests ###
class GenerationJobManagerTests(unittest.TestCase):
    def test_jobs_have_independent_names_stages_and_progress(self) -> None:
        manager = GenerationJobManager()

        first = manager.create_job(
            kind="Object generation",
            requested_name="  Chair  ",
            default_name="Generated object 1",
            stage="Queued",
        )
        second = manager.create_job(
            kind="Surface texture generation",
            requested_name="",
            default_name="Wall texture 2",
            stage="Preparing references",
        )
        manager.update_job(first.job_id, stage="Generating model (37%)")

        self.assertEqual(first.name, "Chair")
        self.assertEqual(second.name, "Wall texture 2")
        self.assertEqual(len(manager.jobs()), 2)
        self.assertEqual(manager.get_job(first.job_id).progress, 37)  # type: ignore[union-attr]
        self.assertEqual(manager.get_job(second.job_id).progress, None)  # type: ignore[union-attr]

    def test_cancel_is_non_reentrant_and_keeps_synchronous_terminal_state(
        self,
    ) -> None:
        manager = GenerationJobManager()
        job = manager.create_job(
            kind="Object generation",
            requested_name="Table",
            default_name="Generated object 1",
            stage="Generating",
        )
        nested_results: list[bool] = []

        def cancel() -> bool:
            nested_results.append(manager.cancel_job(job.job_id))
            manager.mark_cancelled(job.job_id)
            return True

        manager.set_cancel_callback(job.job_id, cancel)

        self.assertTrue(manager.cancel_job(job.job_id))
        self.assertEqual(nested_results, [False])
        self.assertEqual(
            manager.get_job(job.job_id).status,  # type: ignore[union-attr]
            JOB_STATUS_CANCELLED,
        )
        self.assertFalse(manager.cancel_job(job.job_id))

    def test_rejected_cancel_restores_running_presentation(self) -> None:
        manager = GenerationJobManager()
        job = manager.create_job(
            kind="Object texture generation",
            requested_name="",
            default_name="Existing object texture",
            stage="Uploading model (12%)",
        )
        manager.set_cancel_callback(job.job_id, lambda: False)

        self.assertFalse(manager.cancel_job(job.job_id))

        restored = manager.get_job(job.job_id)
        self.assertIsNotNone(restored)
        self.assertEqual(restored.status, JOB_STATUS_RUNNING)  # type: ignore[union-attr]
        self.assertEqual(restored.stage, "Uploading model (12%)")  # type: ignore[union-attr]
        self.assertEqual(restored.progress, 12)  # type: ignore[union-attr]

    def test_stale_progress_cannot_replace_cancelling_stage(self) -> None:
        manager = GenerationJobManager()
        job = manager.create_job(
            kind="Object generation",
            requested_name="Chair",
            default_name="Generated object",
            stage="Generating (40%)",
        )

        def cancel() -> bool:
            manager.update_job(job.job_id, stage="Late provider update (80%)")
            return True

        manager.set_cancel_callback(job.job_id, cancel)
        self.assertTrue(manager.cancel_job(job.job_id))

        cancelling = manager.get_job(job.job_id)
        self.assertEqual(cancelling.stage, "Cancelling...")  # type: ignore[union-attr]
        self.assertEqual(cancelling.progress, 40)  # type: ignore[union-attr]

    def test_finished_job_cannot_regain_a_cancel_callback(self) -> None:
        manager = GenerationJobManager()
        job = manager.create_job(
            kind="Object generation",
            requested_name="",
            default_name="Finished object",
            stage="Queued",
        )
        manager.complete_job(job.job_id)
        manager.set_cancel_callback(job.job_id, lambda: True)

        self.assertNotIn(job.job_id, manager._cancel_callbacks)
        self.assertFalse(manager.cancel_job(job.job_id))

    def test_stage_without_percentage_preserves_known_progress(self) -> None:
        manager = GenerationJobManager()
        job = manager.create_job(
            kind="Object generation",
            requested_name="",
            default_name="Generated object",
            stage="Generating (48%)",
        )

        manager.update_job(job.job_id, stage="Preparing texture variants")

        updated = manager.get_job(job.job_id)
        self.assertIsNotNone(updated)
        self.assertEqual(updated.progress, 48)  # type: ignore[union-attr]

    def test_running_progress_is_monotonic_and_reserves_one_hundred(self) -> None:
        manager = GenerationJobManager()
        job = manager.create_job(
            kind="Object generation",
            requested_name="",
            default_name="Generated object",
            stage="Geometry complete (100%)",
        )

        self.assertEqual(manager.get_job(job.job_id).progress, 99)  # type: ignore[union-attr]
        manager.update_job(job.job_id, stage="Texturing (12%)")
        self.assertEqual(manager.get_job(job.job_id).progress, 99)  # type: ignore[union-attr]

        manager.complete_job(job.job_id, "Committed")
        completed = manager.get_job(job.job_id)
        self.assertEqual(completed.progress, 100)  # type: ignore[union-attr]
        self.assertEqual(completed.status, JOB_STATUS_COMPLETED)  # type: ignore[union-attr]

    def test_active_job_can_be_renamed_and_terminal_state_is_immutable(
        self,
    ) -> None:
        manager = GenerationJobManager()
        job = manager.create_job(
            kind="Object generation",
            requested_name="",
            default_name="Object from frame 1",
            stage="Generating (30%)",
        )

        renamed = manager.rename_job(job.job_id, "Provider chair")
        self.assertEqual(renamed.name, "Provider chair")  # type: ignore[union-attr]
        manager.complete_job(job.job_id, "Committed")
        manager.update_job(job.job_id, stage="Late progress (42%)")
        manager.rename_job(job.job_id, "Late rename")

        completed = manager.get_job(job.job_id)
        self.assertEqual(completed.name, "Provider chair")  # type: ignore[union-attr]
        self.assertEqual(completed.stage, "Committed")  # type: ignore[union-attr]
        self.assertEqual(completed.progress, 100)  # type: ignore[union-attr]


# ### Jobs window tests ###
class JobsWindowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = GenerationJobManager()
        self.parent_widget = QWidget()
        self.window = JobsWindow(self.manager, self.parent_widget)

    def tearDown(self) -> None:
        self.window.dispose()
        self.parent_widget.deleteLater()
        _qt_application.processEvents()

    def test_new_job_opens_window_and_finished_jobs_can_be_cleared(self) -> None:
        job = self.manager.create_job(
            kind="Surface texture generation",
            requested_name="Brick wall",
            default_name="Wall texture",
            stage="Queued",
        )
        _qt_application.processEvents()

        self.assertTrue(self.window.isVisible())
        self.assertTrue(self.window.isMaximized())
        self.assertIn(job.job_id, self.window._rows)
        row = self.window._rows[job.job_id]
        self.assertEqual(row.name_label.text(), "Brick wall")
        self.assertEqual(row.progress_bar.minimum(), 0)
        self.assertEqual(row.progress_bar.maximum(), 0)

        self.manager.update_job(job.job_id, stage="Rendering (66%)")
        self.manager.complete_job(job.job_id)
        _qt_application.processEvents()

        self.assertEqual(row.progress_bar.value(), 100)
        self.assertEqual(
            self.manager.get_job(job.job_id).status,  # type: ignore[union-attr]
            JOB_STATUS_COMPLETED,
        )
        self.assertTrue(self.window.clear_finished_button.isEnabled())

        self.manager.clear_finished()
        _qt_application.processEvents()

        self.assertEqual(self.manager.jobs(), ())
        self.assertEqual(self.window._rows, {})
        self.assertTrue(self.window.empty_label.isVisible())

    def test_user_close_hides_window_until_the_next_job_reopens_it(self) -> None:
        self.manager.create_job(
            kind="Object generation",
            requested_name="First job",
            default_name="Generated object",
            stage="Queued",
        )
        _qt_application.processEvents()
        self.assertTrue(self.window.isVisible())
        self.assertTrue(self.window.isMaximized())

        self.window.showNormal()
        self.window.close()
        _qt_application.processEvents()

        self.manager.create_job(
            kind="Object texture generation",
            requested_name="Third job",
            default_name="Object texture",
            stage="Queued",
        )
        _qt_application.processEvents()
        self.assertTrue(self.window.isVisible())
        self.assertFalse(self.window.isMaximized())

        self.window.close()
        _qt_application.processEvents()
        self.assertFalse(self.window.isVisible())

        self.manager.create_job(
            kind="Surface texture generation",
            requested_name="Second job",
            default_name="Wall texture",
            stage="Queued",
        )
        _qt_application.processEvents()
        self.assertTrue(self.window.isVisible())

    def test_dispose_disconnects_manager_without_detaching_parent(self) -> None:
        parent = self.window.parent()
        known_rows = set(self.window._rows)

        self.window.dispose()
        late_job = self.manager.create_job(
            kind="Object generation",
            requested_name="Too late",
            default_name="Generated object",
            stage="Queued",
        )

        self.assertTrue(self.window._is_disposed)
        self.assertIs(self.window.parent(), parent)
        self.assertFalse(self.window.isVisible())
        self.assertEqual(set(self.window._rows), known_rows)
        self.assertNotIn(late_job.job_id, self.window._rows)

    def test_removed_target_screen_falls_back_to_primary(self) -> None:
        removed_screen = object()
        primary_screen = object()
        self.window._target_screen = removed_screen  # type: ignore[assignment]

        with (
            patch(
                "housemaker.generation_jobs.QGuiApplication.primaryScreen",
                return_value=primary_screen,
            ),
            patch.object(
                self.window,
                "_try_move_to_screen",
                side_effect=(False, True),
            ) as move_mock,
        ):
            self.window._move_to_target_screen()

        self.assertEqual(
            [call.args[0] for call in move_mock.call_args_list],
            [removed_screen, primary_screen],
        )
        self.assertIsNone(self.window._target_screen)


# ### Settings integration tests ###
class JobsWindowSettingsTests(unittest.TestCase):
    def test_display_selection_persists_and_is_available_without_file_io(
        self,
    ) -> None:
        options = (
            Fullscreen3DViewerScreenOption(
                screen_id="monitor:Acme|Panel|002",
                label="Second display (1920 x 1080)",
            ),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            settings_store = ApplicationSettingsStore(
                Path(temporary_directory) / "settings.json"
            )
            with patch(
                "housemaker.settings_widget."
                "connected_fullscreen_3d_viewer_display_options",
                return_value=options,
            ):
                widget = SettingsWidget(
                    application_settings=settings_store,
                    environment={},
                )
                self.addCleanup(widget.deleteLater)
                combo = widget.jobs_window_screen_combo

                self.assertEqual(combo.itemText(0), "Primary display")
                self.assertIsNone(combo.itemData(0))
                combo.setCurrentIndex(1)

                self.assertEqual(
                    settings_store.get(JOBS_WINDOW_SCREEN_SETTING_KEY),
                    options[0].screen_id,
                )
                self.assertEqual(
                    read_jobs_window_screen_id(settings_store),
                    options[0].screen_id,
                )
                with patch.object(
                    settings_store,
                    "get",
                    side_effect=AssertionError("unexpected settings read"),
                ):
                    selected_id = widget.get_jobs_window_screen_id()
                self.assertEqual(selected_id, options[0].screen_id)

    def test_new_setting_keeps_existing_positional_constructor_order(self) -> None:
        settings = GenerationServiceSettings(
            "meshy-key",
            4_000,
            "openai-key",
            "meshy",
            None,
            "Ctrl+N",
            True,
        )

        self.assertEqual(settings.canvas_3d_navigation_toggle_hotkey, "Ctrl+N")
        self.assertTrue(settings.unused_face_removal)
        self.assertIsNone(settings.jobs_window_screen_id)

    def test_synthesized_duplicate_screen_id_resolves_to_matching_screen(
        self,
    ) -> None:
        first_screen = _FakeScreen("Duplicated display")
        second_screen = _FakeScreen("Duplicated display")
        screens = (first_screen, second_screen)

        with patch(
            "housemaker.settings_widget._connected_screens",
            return_value=screens,
        ):
            options = connected_fullscreen_3d_viewer_display_options()
            resolved = resolve_fullscreen_3d_viewer_screen(
                options[1].screen_id
            )

        self.assertEqual(options[0].screen_id, "screen:Duplicated display")
        self.assertEqual(options[1].screen_id, "screen:Duplicated display|2")
        self.assertIs(resolved, second_screen)

    def test_primary_screen_change_signal_refreshes_display_targets(self) -> None:
        screen_application = _FakeScreenApplication()
        with tempfile.TemporaryDirectory() as temporary_directory:
            settings_store = ApplicationSettingsStore(
                Path(temporary_directory) / "settings.json"
            )
            with patch(
                "housemaker.settings_widget._get_gui_application",
                return_value=screen_application,
            ):
                widget = SettingsWidget(
                    application_settings=settings_store,
                    environment={},
                )
                self.addCleanup(widget.deleteLater)
                changes: list[bool] = []
                widget.settings_changed.connect(
                    lambda: changes.append(True)
                )

                screen_application.primaryScreenChanged.emit(None)

        self.assertEqual(changes, [True])

    def test_dispose_disconnects_every_application_screen_signal(self) -> None:
        screen_application = _FakeScreenApplication()
        with tempfile.TemporaryDirectory() as temporary_directory:
            settings_store = ApplicationSettingsStore(
                Path(temporary_directory) / "settings.json"
            )
            with patch(
                "housemaker.settings_widget._get_gui_application",
                return_value=screen_application,
            ):
                widget = SettingsWidget(
                    application_settings=settings_store,
                    environment={},
                )
                changes: list[bool] = []
                widget.settings_changed.connect(
                    lambda: changes.append(True)
                )

                widget.dispose()
                widget.dispose()
                screen_application.screenAdded.emit(None)
                screen_application.screenRemoved.emit(None)
                screen_application.primaryScreenChanged.emit(None)

        self.assertEqual(changes, [])
        self.assertEqual(screen_application.screenAdded.slot_count, 0)
        self.assertEqual(screen_application.screenRemoved.slot_count, 0)
        self.assertEqual(
            screen_application.primaryScreenChanged.slot_count,
            0,
        )
        widget.deleteLater()


# ### Main workspace integration tests ###
class JobsWindowMainIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = BlueprintWorkspace(
            application_settings=ApplicationSettingsStore(
                Path(self._temporary_directory.name) / "settings.json"
            )
        )
        self.workspace.resize(1200, 800)
        self.workspace.show()
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.workspace.close()
        self.workspace.deleteLater()
        _qt_application.processEvents()
        self._temporary_directory.cleanup()

    def test_generation_tabs_share_one_persistent_jobs_window(self) -> None:
        self.assertIs(
            self.workspace.generation._job_manager,
            self.workspace.job_manager,
        )
        self.assertIs(
            self.workspace.surface_texture_generation._job_manager,
            self.workspace.job_manager,
        )

        job = self.workspace.job_manager.create_job(
            kind="Object generation",
            requested_name="Lamp",
            default_name="Generated object",
            stage="Queued",
        )
        _qt_application.processEvents()

        self.assertTrue(self.workspace.jobs_window.isVisible())
        self.assertIn(job.job_id, self.workspace.jobs_window._rows)

    def test_settings_changes_retarget_the_jobs_window(self) -> None:
        selected_screen_id = "screen:jobs-test"
        selected_screen = object()
        combo = self.workspace.settings_widget.jobs_window_screen_combo
        combo.addItem("Jobs test display", selected_screen_id)

        with (
            patch(
                "housemaker.main.resolve_fullscreen_3d_viewer_screen",
                return_value=selected_screen,
            ) as resolve_mock,
            patch.object(
                self.workspace.jobs_window,
                "set_target_screen",
            ) as target_mock,
        ):
            combo.setCurrentIndex(combo.findData(selected_screen_id))

        resolve_mock.assert_called_with(selected_screen_id)
        target_mock.assert_called_with(selected_screen)

    def test_primary_display_topology_change_repositions_visible_window(
        self,
    ) -> None:
        self.workspace.job_manager.create_job(
            kind="Object generation",
            requested_name="Visible job",
            default_name="Generated object",
            stage="Queued",
        )
        _qt_application.processEvents()

        with patch.object(
            self.workspace.jobs_window,
            "_move_to_target_screen",
        ) as move_mock:
            self.workspace.settings_widget._handle_connected_screens_changed(
                None
            )

        move_mock.assert_called_once_with()

    def test_persisted_display_is_applied_during_workspace_startup(self) -> None:
        selected_screen_id = "screen:persisted-jobs-display"
        selected_screen = object()
        with tempfile.TemporaryDirectory() as temporary_directory:
            settings_store = ApplicationSettingsStore(
                Path(temporary_directory) / "settings.json"
            )
            settings_store.set(
                JOBS_WINDOW_SCREEN_SETTING_KEY,
                selected_screen_id,
            )
            options = (
                Fullscreen3DViewerScreenOption(
                    screen_id=selected_screen_id,
                    label="Persisted Jobs display",
                ),
            )
            with (
                patch(
                    "housemaker.settings_widget."
                    "connected_fullscreen_3d_viewer_display_options",
                    return_value=options,
                ),
                patch(
                    "housemaker.main.resolve_fullscreen_3d_viewer_screen",
                    return_value=selected_screen,
                ),
            ):
                workspace = BlueprintWorkspace(
                    application_settings=settings_store
                )
            try:
                self.assertIs(
                    workspace.jobs_window._target_screen,
                    selected_screen,
                )
            finally:
                workspace.shutdown()
                workspace.close()
                workspace.deleteLater()
                _qt_application.processEvents()

    def test_workspace_shutdown_disposes_global_settings_connections_once(
        self,
    ) -> None:
        with patch.object(
            self.workspace.settings_widget,
            "dispose",
            wraps=self.workspace.settings_widget.dispose,
        ) as dispose_mock:
            self.workspace.shutdown()
            self.workspace.shutdown()

        dispose_mock.assert_called_once_with()
        with patch.object(
            self.workspace.generation,
            "set_runtime_settings",
        ) as runtime_settings_mock:
            self.workspace.settings_widget.settings_changed.emit()
        runtime_settings_mock.assert_not_called()


# ### Test helpers ###
class _FakeScreen:
    def __init__(self, name: str) -> None:
        self._name = name

    def name(self) -> str:
        return self._name

    def manufacturer(self) -> str:
        return ""

    def model(self) -> str:
        return ""

    def serialNumber(self) -> str:
        return ""


class _FakeSignal:
    def __init__(self) -> None:
        self._slots = []

    def connect(self, slot) -> None:
        self._slots.append(slot)

    def disconnect(self, slot) -> None:
        try:
            self._slots.remove(slot)
        except ValueError as error:
            raise RuntimeError("Slot is not connected") from error

    def emit(self, value) -> None:
        for slot in tuple(self._slots):
            slot(value)

    @property
    def slot_count(self) -> int:
        return len(self._slots)


class _FakeScreenApplication:
    def __init__(self) -> None:
        self.screenAdded = _FakeSignal()
        self.screenRemoved = _FakeSignal()
        self.primaryScreenChanged = _FakeSignal()

    def screens(self) -> list[object]:
        return []


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
