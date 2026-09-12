# ### Environment setup ###
from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.atlas_export import AtlasDrawCallEstimate
from housemaker.main import BlueprintWorkspace

# ### Test application ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Helpers ###
def _wait_until(predicate, timeout_seconds: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        _qt_application.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    _qt_application.processEvents()
    if not predicate():
        raise AssertionError("Timed out waiting for the draw-call estimate.")


# ### Main integration tests ###
class AtlasDrawCallMainIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        settings = ApplicationSettingsStore(
            Path(self._temporary_directory.name) / "settings.json"
        )
        self.workspace = BlueprintWorkspace(application_settings=settings)
        self.workspace.show()
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        self.workspace.deleteLater()
        _qt_application.processEvents()
        self._temporary_directory.cleanup()

    def test_opening_atlas_calculates_empty_scene_without_blocking(self) -> None:
        self.assertIsNone(self.workspace.texture_atlas_workspace.draw_call_estimate)

        self.workspace.workspace_tabs.setCurrentWidget(
            self.workspace.texture_atlas_workspace
        )
        _wait_until(
            lambda: (
                self.workspace.texture_atlas_workspace.draw_call_estimate is not None
            )
        )

        self.assertEqual(
            self.workspace.texture_atlas_workspace.draw_call_estimate,
            AtlasDrawCallEstimate(),
        )
        self.assertEqual(
            self.workspace.texture_atlas_workspace.draw_call_estimate_value_label.text(),
            "0",
        )

    def test_reopening_unchanged_atlas_reuses_revision_cache(self) -> None:
        self.workspace.workspace_tabs.setCurrentWidget(
            self.workspace.texture_atlas_workspace
        )
        _wait_until(
            lambda: (
                self.workspace.texture_atlas_workspace.draw_call_estimate is not None
            )
        )

        self.workspace.workspace_tabs.setCurrentWidget(
            self.workspace.canvas_viewer_workspace
        )
        with patch("housemaker.main._AtlasDrawCallEstimateThread") as thread_type:
            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.texture_atlas_workspace
            )
            _wait_until(
                lambda: (
                    self.workspace.texture_atlas_workspace.draw_call_estimate_value_label.text()
                    == "0"
                )
            )

        thread_type.assert_not_called()

    def test_estimator_failure_is_explained_in_tooltip(self) -> None:
        def failure_is_visible() -> bool:
            atlas_workspace = self.workspace.texture_atlas_workspace
            tooltip = atlas_workspace.draw_call_estimate_value_label.toolTip()
            return "corrupt placed object" in tooltip

        with patch(
            "housemaker.main._build_surface_ao_pre_atlas_scene",
            side_effect=ValueError("corrupt placed object"),
        ):
            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.texture_atlas_workspace
            )
            _wait_until(failure_is_visible)

        self.assertEqual(
            self.workspace.texture_atlas_workspace.draw_call_estimate_value_label.text(),
            "Unavailable",
        )

    def test_transform_only_preview_change_keeps_draw_call_revision(self) -> None:
        revision_before = self.workspace._atlas_draw_call_scene_revision

        with patch.object(
            self.workspace,
            "_schedule_atlas_draw_call_estimate",
        ) as schedule:
            self.workspace._mark_viewer_preview_dirty(affects_draw_call_estimate=False)

        self.assertEqual(
            self.workspace._atlas_draw_call_scene_revision,
            revision_before,
        )
        schedule.assert_not_called()


if __name__ == "__main__":
    unittest.main()
