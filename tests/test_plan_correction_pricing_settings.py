# ### Environment setup ###
from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PySide6.QtCore import QObject, QUrl, Signal
from PySide6.QtNetwork import QNetworkReply, QNetworkRequest
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.openai_model_pricing import (
    LOCAL_MODEL_PRICE_TEXT,
    OPENAI_PRICING_RESPONSE_LIMIT_BYTES,
    OPENAI_PRICING_TIMEOUT_MILLISECONDS,
    PRICE_NOT_FOUND_TEXT,
)
from housemaker.plan_correction_models import (
    PLAN_CORRECTION_MODEL_GPT_5_6_LUNA,
    PLAN_CORRECTION_MODEL_GPT_5_6_TERRA,
    PLAN_CORRECTION_MODEL_GPT_IMAGE_2,
    PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1,
)
from housemaker.settings_widget import (
    PLAN_CORRECTION_MODEL_SETTING_KEY,
    SettingsWidget,
)

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Test doubles ###
class _FakePricingReply(QObject):
    readyRead = Signal()
    downloadProgress = Signal(int, int)
    finished = Signal()

    def __init__(
        self,
        *,
        status: int = 200,
        content_type: str = "text/markdown; charset=utf-8",
        url: str = "https://developers.openai.com/api/docs/pricing.md",
        error: QNetworkReply.NetworkError = (
            QNetworkReply.NetworkError.NoError
        ),
    ) -> None:
        super().__init__()
        self._status = status
        self._content_type = content_type
        self._url = QUrl(url)
        self._error = error
        self._buffer = bytearray()
        self.aborted = False
        self.deleted = False

    def feed(self, payload: bytes, *, total: int | None = None) -> None:
        self._buffer.extend(payload)
        advertised_total = len(payload) if total is None else total
        self.downloadProgress.emit(len(payload), advertised_total)
        self.readyRead.emit()

    def readAll(self) -> bytes:
        payload = bytes(self._buffer)
        self._buffer.clear()
        return payload

    def error(self) -> QNetworkReply.NetworkError:
        return self._error

    def attribute(self, attribute: QNetworkRequest.Attribute) -> object:
        if attribute == QNetworkRequest.Attribute.HttpStatusCodeAttribute:
            return self._status
        return None

    def header(self, header: QNetworkRequest.KnownHeaders) -> object:
        if header == QNetworkRequest.KnownHeaders.ContentTypeHeader:
            return self._content_type
        return None

    def url(self) -> QUrl:
        return QUrl(self._url)

    def abort(self) -> None:
        self.aborted = True

    def deleteLater(self) -> None:
        self.deleted = True


class _FakePricingNetworkManager:
    def __init__(self, reply: _FakePricingReply) -> None:
        self.reply = reply
        self.request: QNetworkRequest | None = None

    def get(self, request: QNetworkRequest) -> _FakePricingReply:
        self.request = request
        return self.reply


# ### Pricing fixture ###
def _pricing_markdown() -> bytes:
    return b"""# Pricing
Flagship models
Standard
### Standard pricing data
| Model | Short context input | Short context cached input | Short context cache writes | Short context output | Long context input | Long context cached input | Long context cache writes | Long context output |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-5.6-terra | $2.00 | $0.20 | $2.50 | $12.00 | $4.00 | $0.40 | $5.00 | $18.00 |
| gpt-5.6-luna | $0.20 | $0.02 | $0.25 | $1.20 | $0.40 | $0.04 | $0.50 | $1.80 |
Cyber models
Image generation models
Standard
### Grouped Pricing Table data
| Model | Modality | Input | Cached input | Output |
| --- | --- | --- | --- | --- |
| gpt-image-2 | Image | $8.00 | $2.00 | $30.00 |
| gpt-image-2 | Text | $5.00 | $1.25 | - |
Batch
Video generation models
"""


# ### Widget tests ###
def test_successful_lookup_relabels_without_changing_selection_or_settings() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        store = _build_test_settings(temporary_directory)
        store.set(
            PLAN_CORRECTION_MODEL_SETTING_KEY,
            PLAN_CORRECTION_MODEL_GPT_5_6_LUNA,
        )
        reply = _FakePricingReply()
        manager = _FakePricingNetworkManager(reply)
        widget = SettingsWidget(
            application_settings=store,
            environment={},
            plan_pricing_network_manager=manager,  # type: ignore[arg-type]
        )
        emitted_changes: list[bool] = []
        widget.settings_changed.connect(lambda: emitted_changes.append(True))

        widget.show()
        _qt_application.processEvents()
        assert manager.request is not None
        assert (
            manager.request.transferTimeout()
            == OPENAI_PRICING_TIMEOUT_MILLISECONDS
        )
        assert manager.request.attribute(
            QNetworkRequest.Attribute.RedirectPolicyAttribute
        ) == QNetworkRequest.RedirectPolicy.NoLessSafeRedirectPolicy
        reply.feed(_pricing_markdown())
        reply.finished.emit()

        combo = widget.plan_correction_model_combo
        assert combo.currentData() == PLAN_CORRECTION_MODEL_GPT_5_6_LUNA
        assert store.get(PLAN_CORRECTION_MODEL_SETTING_KEY) == (
            PLAN_CORRECTION_MODEL_GPT_5_6_LUNA
        )
        assert emitted_changes == []
        assert _item_text(combo, PLAN_CORRECTION_MODEL_GPT_IMAGE_2) == (
            "gpt-image-2 ($5.00 text in, $8.00 image in / "
            "$30.00 image out per 1M tokens)"
        )
        assert _item_text(combo, PLAN_CORRECTION_MODEL_GPT_5_6_LUNA) == (
            "GPT-5.6 Luna ($0.20 in / $1.20 out per 1M tokens)"
        )
        assert _item_text(combo, PLAN_CORRECTION_MODEL_GPT_5_6_TERRA) == (
            "GPT-5.6 Terra ($2.00 in / $12.00 out per 1M tokens)"
        )
        assert _item_text(combo, PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1) == (
            f"Qwen-Image-2.1 ({LOCAL_MODEL_PRICE_TEXT})"
        )
        assert reply.deleted
        widget.dispose()


def test_failed_or_untrusted_response_keeps_exact_price_fallback() -> None:
    untrusted_urls = (
        "https://example.com/pricing.md",
        "https://developers.openai.com/api/docs/models.md",
    )
    for untrusted_url in untrusted_urls:
        with tempfile.TemporaryDirectory() as temporary_directory:
            reply = _FakePricingReply(url=untrusted_url)
            manager = _FakePricingNetworkManager(reply)
            widget = SettingsWidget(
                application_settings=_build_test_settings(
                    temporary_directory
                ),
                environment={},
                plan_pricing_network_manager=manager,  # type: ignore[arg-type]
            )

            widget._start_plan_correction_pricing_lookup()
            reply.feed(_pricing_markdown())
            reply.finished.emit()

            combo = widget.plan_correction_model_combo
            for model_id in (
                PLAN_CORRECTION_MODEL_GPT_IMAGE_2,
                PLAN_CORRECTION_MODEL_GPT_5_6_LUNA,
                PLAN_CORRECTION_MODEL_GPT_5_6_TERRA,
            ):
                assert _item_text(combo, model_id).endswith(
                    f"({PRICE_NOT_FOUND_TEXT})"
                )
            widget.dispose()


def test_oversized_response_is_aborted_and_not_parsed() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        reply = _FakePricingReply()
        manager = _FakePricingNetworkManager(reply)
        widget = SettingsWidget(
            application_settings=_build_test_settings(temporary_directory),
            environment={},
            plan_pricing_network_manager=manager,  # type: ignore[arg-type]
        )

        widget._start_plan_correction_pricing_lookup()
        reply.feed(b"x" * (OPENAI_PRICING_RESPONSE_LIMIT_BYTES + 1))
        assert reply.aborted
        reply.finished.emit()

        assert _item_text(
            widget.plan_correction_model_combo,
            PLAN_CORRECTION_MODEL_GPT_IMAGE_2,
        ).endswith(f"({PRICE_NOT_FOUND_TEXT})")
        widget.dispose()


def test_dispose_aborts_and_detaches_an_active_lookup() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        reply = _FakePricingReply()
        manager = _FakePricingNetworkManager(reply)
        widget = SettingsWidget(
            application_settings=_build_test_settings(temporary_directory),
            environment={},
            plan_pricing_network_manager=manager,  # type: ignore[arg-type]
        )

        widget._start_plan_correction_pricing_lookup()
        widget.dispose()

        assert reply.aborted
        assert reply.deleted
        reply.finished.emit()


# ### Test helpers ###
def _build_test_settings(directory: str) -> ApplicationSettingsStore:
    return ApplicationSettingsStore(Path(directory) / "settings.json")


def _item_text(combo: object, model_id: str) -> str:
    index = combo.findData(model_id)  # type: ignore[attr-defined]
    assert index >= 0
    return str(combo.itemText(index))  # type: ignore[attr-defined]
