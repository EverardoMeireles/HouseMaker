"""Isolated Reflect3D worker: never import Torch inside the editor process."""

# ### Imports ###
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import importlib.util
import json
import os
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace


# ### Verified upstream symmetry decoder loading ###
def _load_decoder(source: Path):
    from housemaker.reflect3d_reference_setup import REFLECT3D_SOURCE_SHA256

    for relative, expected in REFLECT3D_SOURCE_SHA256.items():
        if hashlib.sha256((source / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Reflect3D source checksum mismatch: {relative}")
    sys.path.insert(0, str(source))
    # Only this unused point-cloud import pulls in trimesh/plyfile. Reflect3D's
    # feed-forward decoder uses normalize=False, so shift_scale_points is never
    # called. Keep the pinned original bytes and remove the import in memory.
    name = "model.threedetr.position_embedding"
    path = source / "model" / "threedetr" / "position_embedding.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and node.module == "utils.pc_util"
    )]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(tree, str(path), "exec"), module.__dict__)  # noqa: S102 - checksum-pinned upstream source.
    from model.shape.symm_decoder import SymmDecoder

    # Upstream hard-codes .cuda() for a scalar clamp. Device-neutral arithmetic
    # preserves the operation and also makes CPU fallback/verification possible.
    def normalize_vector(_self, vector, return_mag=False):
        magnitude = vector.square().sum(dim=1).sqrt().clamp_min(1e-8)
        normalized = vector / magnitude[:, None]
        return (normalized, magnitude) if return_mag else normalized

    SymmDecoder.normalize_vector = normalize_vector
    return SymmDecoder


def load_reflect3d_model(directory: Path, *, device: str):
    """Build just the pretrained DINOv2 encoder and reflection decoder offline."""
    import numpy as np
    import torch
    from torch import nn
    from transformers import Dinov2Config, Dinov2Model

    from housemaker.reflect3d_reference_setup import REFLECT3D_WEIGHTS_FILENAME

    source = directory / "source" / "feedforward_symm"
    decoder_type = _load_decoder(source)
    queries = torch.from_numpy(np.asarray(json.loads(
        (source / "data" / "symm" / "candidate_planes.json").read_text(encoding="utf-8"),
    ), dtype=np.float32))
    if queries.shape != (31, 3):
        raise ValueError("Reflect3D's pinned query-normal table has an invalid shape.")
    options = SimpleNamespace(
        arch=SimpleNamespace(latent_dim=1024, impl=SimpleNamespace(n_channels=256)),
        symm_decoder=SimpleNamespace(dim=256, num_head=4, ffn_dim=256, dropout=0.1, num_layers=4),
    )
    # These are facebook/dinov2-large's published config values. The official
    # Reflect3D checkpoint already includes ALL encoder weights: no HF download,
    # depth model, reconstruction network, custom CUDA extension, or diffusion.
    config = Dinov2Config(
        hidden_size=1024, num_hidden_layers=24, num_attention_heads=16,
        image_size=518, patch_size=14, layerscale_value=1.0,
        layer_norm_eps=1e-6, mlp_ratio=4, use_swiglu_ffn=False,
    )
    config._attn_implementation = "eager"

    class FeedForwardSymmetry(nn.Module):
        def __init__(self):
            super().__init__()
            self.dino_model = Dinov2Model(config)
            self.symm_decoder = decoder_type(options, queries)
            self.pos_embed = nn.Parameter(torch.zeros(1, 257, 256), requires_grad=False)

        def forward(self, pixels):
            features = self.dino_model(pixels).last_hidden_state
            output = self.symm_decoder(features, pre_computed_pos_emb=True, enc_pos=self.pos_embed)["outputs"]
            return output["normal_normalized"], output["cls_logits"].sigmoid()

    # Meta construction avoids initializing a second 1.2GB random encoder.
    with torch.device("meta"):
        model = FeedForwardSymmetry()
    checkpoint = torch.load(
        directory / REFLECT3D_WEIGHTS_FILENAME, map_location="cpu", weights_only=True, mmap=True,
    )
    model.load_state_dict(checkpoint["graph"], strict=True, assign=True)
    del checkpoint
    return model.eval().to(device)


# ### Small JSON-line inference protocol ###
def _emit(message: dict) -> None:
    print(json.dumps(message, separators=(",", ":"), allow_nan=False), flush=True)


def run(directory: Path) -> None:
    import numpy as np
    import torch

    from housemaker.reflect3d_reference_symmetry import REFLECT3D_INPUT_SIZE

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    device = os.environ.get("HOUSEMAKER_REFLECT3D_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    with redirect_stdout(sys.stderr):
        model = load_reflect3d_model(directory, device=device)
    if device == "cuda":
        torch.cuda.synchronize()
    _emit({"event": "ready", "device": device, "startup_seconds": time.perf_counter() - started})
    expected_bytes = REFLECT3D_INPUT_SIZE * REFLECT3D_INPUT_SIZE * 3
    for line in sys.stdin:
        request = json.loads(line)
        if not isinstance(request, dict) or type(request.get("id")) is not int:
            raise ValueError("Invalid Reflect3D request identifier.")
        encoded = request.get("pixels")
        if not isinstance(encoded, str) or len(encoded) > (expected_bytes + 2) // 3 * 4:
            raise ValueError("Invalid Reflect3D image payload.")
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) != expected_bytes:
            raise ValueError("Reflect3D image payload has the wrong dimensions.")
        pixels = np.frombuffer(raw, np.uint8).reshape(REFLECT3D_INPUT_SIZE, REFLECT3D_INPUT_SIZE, 3).copy()
        started = time.perf_counter()
        tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        mean = tensor.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = tensor.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        with torch.inference_mode():
            normals, scores = model((tensor - mean) / std)
        normals = normals[0].float().cpu().numpy().tolist()
        probabilities = scores[0, :, 0].float().cpu().numpy().tolist()
        _emit({
            "event": "result", "id": request["id"], "normals": normals,
            "probabilities": probabilities, "inference_seconds": time.perf_counter() - started,
        })


# ### Subprocess entry point ###
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.bundle)
    except Exception as exc:
        _emit({"event": "error", "error": f"{type(exc).__name__}: {exc}"[:1000]})
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
