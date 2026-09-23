# ### Environment setup ###
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PySide6.QtWidgets import QApplication, QFormLayout

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.openai_model_pricing import format_plan_correction_model_label
from housemaker.plan_correction_models import (
    DEFAULT_PLAN_CORRECTION_MODEL,
    PLAN_CORRECTION_MODEL_GPT_5_6_LUNA,
    PLAN_CORRECTION_MODEL_GPT_5_6_TERRA,
    PLAN_CORRECTION_MODEL_GPT_IMAGE_2,
    PLAN_CORRECTION_MODEL_OPTIONS,
    PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1,
)
from housemaker.settings_widget import (
    DEFAULT_MAKE_WALLS_CONTINUOUS,
    MAKE_WALLS_CONTINUOUS_SETTING_KEY,
    PLAN_CORRECTION_MODEL_SETTING_KEY,
    GenerationServiceSettings,
    SettingsWidget,
    read_make_walls_continuous,
    read_plan_correction_model,
)

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Test cases ###
class PlanCorrectionModelSettingsTests(unittest.TestCase):
    def test_model_contract_defaults_to_gpt_image_2(self) -> None:
        self.assertEqual(
            PLAN_CORRECTION_MODEL_OPTIONS,
            (
                ("gpt-image-2", "gpt-image-2"),
                ("GPT-5.6 Luna", "gpt-5.6-luna"),
                ("GPT-5.6 Terra", "gpt-5.6-terra"),
                ("Qwen-Image-2.1", "Qwen-Image-2.1"),
            ),
        )
        self.assertEqual(
            DEFAULT_PLAN_CORRECTION_MODEL,
            PLAN_CORRECTION_MODEL_GPT_IMAGE_2,
        )
        self.assertEqual(
            GenerationServiceSettings().plan_correction_model,
            PLAN_CORRECTION_MODEL_GPT_IMAGE_2,
        )
        self.assertEqual(
            GenerationServiceSettings().make_walls_continuous,
            DEFAULT_MAKE_WALLS_CONTINUOUS,
        )

    def test_reader_rejects_stale_or_malformed_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = _build_test_settings(temporary_directory)
            self.assertEqual(
                read_plan_correction_model(store),
                PLAN_CORRECTION_MODEL_GPT_IMAGE_2,
            )

            for value in ("old-model", "", 7, None, ["gpt-image-2"]):
                with self.subTest(value=value):
                    store.set(PLAN_CORRECTION_MODEL_SETTING_KEY, value)
                    self.assertEqual(
                        read_plan_correction_model(store),
                        PLAN_CORRECTION_MODEL_GPT_IMAGE_2,
                    )

            self.assertFalse(read_make_walls_continuous(store))
            for value in (1, "true", None, [], {}):
                with self.subTest(wall_continuity=value):
                    store.set(MAKE_WALLS_CONTINUOUS_SETTING_KEY, value)
                    self.assertFalse(read_make_walls_continuous(store))

    def test_data_model_rejects_unknown_models(self) -> None:
        for value in ("old-model", "", 7, None, ["gpt-image-2"]):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError,
                "Unknown plan correction model",
            ):
                GenerationServiceSettings(
                    plan_correction_model=value,  # type: ignore[arg-type]
                )

        for value in (1, "true", None, [], {}):
            with self.subTest(wall_continuity=value), self.assertRaisesRegex(
                ValueError,
                "Make walls continuous",
            ):
                GenerationServiceSettings(
                    make_walls_continuous=value,  # type: ignore[arg-type]
                )

    def test_dropdown_persists_qwen_selection_and_emits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = _build_test_settings(temporary_directory)
            widget = SettingsWidget(
                application_settings=store,
                environment={},
            )
            emitted_changes: list[bool] = []
            widget.settings_changed.connect(
                lambda: emitted_changes.append(True)
            )
            combo = widget.plan_correction_model_combo
            continuity_checkbox = widget.make_walls_continuous_checkbox

            self.assertEqual(
                tuple(
                    (combo.itemText(index), combo.itemData(index))
                    for index in range(combo.count())
                ),
                tuple(
                    (
                        format_plan_correction_model_label(
                            label,
                            model_id,
                            None,
                        ),
                        model_id,
                    )
                    for label, model_id in PLAN_CORRECTION_MODEL_OPTIONS
                ),
            )
            self.assertEqual(
                combo.currentData(),
                PLAN_CORRECTION_MODEL_GPT_IMAGE_2,
            )
            form = widget.canvas_settings_group.layout()
            self.assertIsInstance(form, QFormLayout)
            assert isinstance(form, QFormLayout)
            self.assertEqual(
                form.labelForField(combo).text(),
                "Plan correction model",
            )
            self.assertEqual(
                form.labelForField(continuity_checkbox).text(),
                "Make walls continuous",
            )
            model_row, _model_role = form.getWidgetPosition(combo)
            continuity_row, _continuity_role = form.getWidgetPosition(
                continuity_checkbox
            )
            self.assertEqual(continuity_row, model_row + 1)
            self.assertFalse(continuity_checkbox.isChecked())

            combo.setCurrentIndex(
                combo.findData(PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1)
            )
            continuity_checkbox.setChecked(True)

            self.assertEqual(
                store.get(PLAN_CORRECTION_MODEL_SETTING_KEY),
                PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1,
            )
            self.assertEqual(
                widget.get_settings().plan_correction_model,
                PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1,
            )
            self.assertTrue(
                store.get(MAKE_WALLS_CONTINUOUS_SETTING_KEY)
            )
            self.assertTrue(
                widget.get_settings().make_walls_continuous
            )
            self.assertEqual(emitted_changes, [True, True])

            for model_id in (
                PLAN_CORRECTION_MODEL_GPT_5_6_LUNA,
                PLAN_CORRECTION_MODEL_GPT_5_6_TERRA,
            ):
                self.assertGreaterEqual(combo.findData(model_id), 0)

            restored = SettingsWidget(
                application_settings=_build_test_settings(
                    temporary_directory
                ),
                environment={},
            )
            self.assertEqual(
                restored.plan_correction_model_combo.currentData(),
                PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1,
            )
            self.assertEqual(
                restored.get_settings().plan_correction_model,
                PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1,
            )
            self.assertTrue(
                restored.make_walls_continuous_checkbox.isChecked()
            )
            self.assertTrue(
                restored.get_settings().make_walls_continuous
            )
            widget.dispose()
            restored.dispose()


# ### Test helpers ###
def _build_test_settings(directory: str) -> ApplicationSettingsStore:
    return ApplicationSettingsStore(Path(directory) / "settings.json")


if __name__ == "__main__":
    unittest.main()
