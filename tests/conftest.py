"""Keep tests away from the application's temporary web-project export folder."""

# ### Imports ###
import pytest


# ### Filesystem isolation ###
@pytest.fixture(autouse=True)
def isolated_runtime_export_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "housemaker.room_export_workspace.RUNTIME_EXPORT_PUBLIC_DIRECTORY",
        tmp_path / "public",
    )
