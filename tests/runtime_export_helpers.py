"""Mock only material extraction in UI tests that use opaque model sentinels."""

# ### Imports ###
from types import SimpleNamespace
from unittest.mock import patch


# ### Export seams ###
def mock_material_export(workspace, model):
    return patch.object(
        workspace,
        "_prepare_export_material_packs",
        return_value=(SimpleNamespace(stripped_model=model, packs=()), {}),
    )
