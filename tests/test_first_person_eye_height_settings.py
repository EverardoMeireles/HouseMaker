"""Persist and validate the physical FPS camera eye height independently."""

# ### Environment and imports ###
from __future__ import annotations

import os
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.first_person_navigation import (
    DEFAULT_FIRST_PERSON_EYE_HEIGHT_METERS,
    FIRST_PERSON_EYE_HEIGHT_STEP_METERS,
    FIRST_PERSON_NAVIGATION_MODE_NOCLIP,
    MAX_FIRST_PERSON_EYE_HEIGHT_METERS,
    MIN_FIRST_PERSON_EYE_HEIGHT_METERS,
    normalize_first_person_eye_height_meters,
)
from housemaker.settings_widget import (
    FIRST_PERSON_EYE_HEIGHT_SETTING_KEY,
    FIRST_PERSON_NAVIGATION_MODE_SETTING_KEY,
    FIRST_PERSON_SPEED_SETTING_KEY,
    GenerationServiceSettings,
    SettingsWidget,
    read_first_person_eye_height,
)

# ### Isolated Qt settings fixture ###
_application = QApplication.instance() or QApplication([])


@pytest.fixture
def widget(tmp_path):
    settings = ApplicationSettingsStore(tmp_path / "preferences.json")
    panel = SettingsWidget(application_settings=settings, environment={})
    try:
        yield panel, settings
    finally:
        panel.dispose()
        panel.close()
        panel.deleteLater()
        _application.processEvents()


# ### Default, controls, persistence and navigation independence ###
def test_default_eye_height_is_170_centimeters_without_rewriting_older_preferences(widget):
    panel, settings = widget
    spin = panel.first_person_eye_height_spinbox

    assert DEFAULT_FIRST_PERSON_EYE_HEIGHT_METERS == 1.70
    assert FIRST_PERSON_EYE_HEIGHT_SETTING_KEY == "navigation/first_person_eye_height_meters"
    assert GenerationServiceSettings().first_person_eye_height_meters == 1.70
    assert panel.get_settings().first_person_eye_height_meters == 1.70
    assert spin.objectName() == "first_person_eye_height_spinbox"
    assert spin.value() == 1.70
    assert spin.decimals() == 2
    assert spin.suffix() == " m"
    assert spin.minimum() == MIN_FIRST_PERSON_EYE_HEIGHT_METERS
    assert spin.maximum() == MAX_FIRST_PERSON_EYE_HEIGHT_METERS
    assert spin.singleStep() == FIRST_PERSON_EYE_HEIGHT_STEP_METERS
    assert not spin.keyboardTracking()
    form = spin.parentWidget().layout()
    assert form.labelForField(spin).text() == "Fps camera eye height"
    assert form.getWidgetPosition(spin)[0] == form.getWidgetPosition(panel.first_person_speed_control)[0] + 1
    assert settings.get(FIRST_PERSON_EYE_HEIGHT_SETTING_KEY) is None


def test_eye_height_persists_emits_once_and_does_not_change_navigation_mode_or_speed(widget):
    panel, settings = widget
    panel.first_person_navigation_combo.setCurrentIndex(
        panel.first_person_navigation_combo.findData(FIRST_PERSON_NAVIGATION_MODE_NOCLIP)
    )
    panel.first_person_speed_slider.setValue(47)
    changed = QSignalSpy(panel.settings_changed)

    panel.first_person_eye_height_spinbox.setValue(1.93)

    assert changed.count() == 1
    assert settings.get(FIRST_PERSON_EYE_HEIGHT_SETTING_KEY) == 1.93
    assert panel.get_settings().first_person_eye_height_meters == 1.93
    assert settings.get(FIRST_PERSON_NAVIGATION_MODE_SETTING_KEY) == FIRST_PERSON_NAVIGATION_MODE_NOCLIP
    assert panel.get_settings().first_person_navigation_mode == FIRST_PERSON_NAVIGATION_MODE_NOCLIP
    assert settings.get(FIRST_PERSON_SPEED_SETTING_KEY) == 4.7
    assert panel.get_settings().first_person_speed_meters_per_second == 4.7
    restored = SettingsWidget(
        application_settings=ApplicationSettingsStore(settings.path), environment={},
    )
    try:
        assert restored.first_person_eye_height_spinbox.value() == 1.93
        assert restored.get_settings().first_person_eye_height_meters == 1.93
        assert restored.get_settings().first_person_navigation_mode == FIRST_PERSON_NAVIGATION_MODE_NOCLIP
        assert restored.get_settings().first_person_speed_meters_per_second == 4.7
    finally:
        restored.dispose()
        restored.close()


# ### Shared model normalization and malformed saved values ###
@pytest.mark.parametrize("raw, expected", ((0.1, 0.1), (1.7, 1.7), (2, 2.0), (10, 10.0), (1.234, 1.23)))
def test_finite_supported_eye_heights_are_normalized_to_centimeters(raw, expected):
    assert normalize_first_person_eye_height_meters(raw) == expected
    assert GenerationServiceSettings(first_person_eye_height_meters=raw).first_person_eye_height_meters == expected


@pytest.mark.parametrize("value", (True, False, None, "1.7", "tall", -1, 0, 0.09, 10.01, float("nan"), float("inf"), float("-inf"), 10**1000))
def test_invalid_eye_heights_are_rejected_by_the_model_and_defaulted_when_loading(value):
    with pytest.raises((TypeError, ValueError)):
        normalize_first_person_eye_height_meters(value)
    with pytest.raises(ValueError, match="First-person eye height"):
        GenerationServiceSettings(first_person_eye_height_meters=value)
    assert read_first_person_eye_height(Mock(get=Mock(return_value=value))) == 1.70


def test_missing_legacy_eye_height_uses_the_default(tmp_path):
    store = ApplicationSettingsStore(tmp_path / "old-preferences.json")
    store.set(FIRST_PERSON_NAVIGATION_MODE_SETTING_KEY, FIRST_PERSON_NAVIGATION_MODE_NOCLIP)

    assert read_first_person_eye_height(store) == 1.70
    assert store.get(FIRST_PERSON_NAVIGATION_MODE_SETTING_KEY) == FIRST_PERSON_NAVIGATION_MODE_NOCLIP
