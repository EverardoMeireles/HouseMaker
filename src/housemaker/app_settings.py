# ### Imports ###
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from PySide6.QtCore import QStandardPaths


# ### Constants ###
APPLICATION_SETTINGS_DIRECTORY_NAME = "HouseMaker"
APPLICATION_SETTINGS_FILE_NAME = "settings.json"


# ### Settings store ###
class ApplicationSettingsStore:
    """Small JSON-backed store for application preferences.

    Reads deliberately fall back to an empty settings object when the file is
    absent or malformed. Writes replace the destination only after the complete
    JSON payload has been written to a temporary file in the same directory.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = (
            default_application_settings_path()
            if path is None
            else Path(path).expanduser()
        )

    def get(self, key: str, default: Any = None) -> Any:
        return self._read().get(str(key), default)

    def set(self, key: str, value: Any) -> bool:
        settings = self._read()
        settings[str(key)] = value
        return self._write(settings)

    def remove(self, key: str) -> bool:
        settings = self._read()
        normalized_key = str(key)
        if normalized_key not in settings:
            return True
        del settings[normalized_key]
        return self._write(settings)

    def _read(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        return {str(key): value for key, value in payload.items()}

    def _write(self, settings: dict[str, Any]) -> bool:
        try:
            serialized_settings = json.dumps(
                settings,
                indent=2,
                sort_keys=True,
            )
        except (TypeError, ValueError, OverflowError):
            return False

        temporary_path: Path | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            file_descriptor, raw_temporary_path = tempfile.mkstemp(
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                dir=str(self.path.parent),
            )
            temporary_path = Path(raw_temporary_path)
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
                handle.write(serialized_settings)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.path)
        except OSError:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass
            return False
        return True


# ### Path helpers ###
def default_application_settings_path() -> Path:
    config_directory = QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.GenericConfigLocation
    )
    if config_directory:
        root_directory = Path(config_directory)
    else:
        root_directory = Path.home() / ".config"
    return (
        root_directory
        / APPLICATION_SETTINGS_DIRECTORY_NAME
        / APPLICATION_SETTINGS_FILE_NAME
    )
