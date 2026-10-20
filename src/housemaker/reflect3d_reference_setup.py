"""Prepare the optional Reflect3D model for generated-reference symmetry checks."""

# ### Imports ###
from __future__ import annotations

import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
import venv
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

# ### Official feed-forward assets and runtime identity ###
REFLECT3D_MODEL_ID = "reflect3d-local"
REFLECT3D_MODEL_NAME = "Reflect3D (local symmetry)"
REFLECT3D_VERSION = 1
REFLECT3D_REVISION = "2a35101fd0f2e4a6f166b0ea8f807f6af1337e72"
REFLECT3D_WEIGHTS_FILENAME = "reflect3d.ckpt"
REFLECT3D_WEIGHTS_SHA256 = "9c248259480b685189b63668eee01a61d7de1d87af96400342ae62a5033033e3"
REFLECT3D_WEIGHTS_URL = (
    "https://drive.usercontent.google.com/download?"
    "id=1rXTDZzV4-TrB36X1Ny9yu7ljCcV-ntWn&export=download&confirm=t"
)
REFLECT3D_SOURCE_ROOT = (
    f"https://raw.githubusercontent.com/RyanXLi/SymmetryStrikesBack/{REFLECT3D_REVISION}/"
    "feedforward_symm/"
)
REFLECT3D_SOURCE_SHA256 = {
    "model/shape/symm_decoder.py": "f7899d04124c6e7a2ae5d09462a1a1aef8875a9496594d70926b63d40d983ee9",
    "model/threedetr/helpers.py": "2ede87a08bc311d5120fff4be719292b8a0b246be308a5e75f019602d8087bd8",
    "model/threedetr/position_embedding.py": "e61c9ca85a7ecaaab084e5fb72a8d23af31fcfd495745d51ae37c49b6a0f6039",
    "model/threedetr/transformer.py": "e95cc90549ddff23d8e620d576b1c2fbb9adf11c1b2b4b6801bc15017a52d28d",
    "data/symm/candidate_planes.json": "773348c564083a6ba07d55d92711632d1996878e68f04c0652495f08b0145d5b",
}
REFLECT3D_TRANSFORMERS_VERSION = "4.46.3"
REFLECT3D_TORCH_VERSION = "2.6.0"
REFLECT3D_RUNTIME_PACKAGES = (
    f"transformers=={REFLECT3D_TRANSFORMERS_VERSION}", "numpy==2.2.6", "Pillow==12.3.0",
)
REFLECT3D_NOTICE = """Reflect3D / Symmetry Strikes Back (CVPR 2025)
Authors: Xiang Li, Zixuan Huang, Anh Thai, James M. Rehg.
Source: https://github.com/RyanXLi/SymmetryStrikesBack
The upstream Reflect3D repository does not currently specify a license.
No license or commercial redistribution permission is inferred by HouseMaker.
Obtain the authors' permission before distributing its code or checkpoint.
The downloaded source retains its original authorship and copyright notices.
Its 3DETR-derived files carry Facebook copyright notices (Apache-2.0 upstream).
DINOv2 is Apache-2.0: https://github.com/facebookresearch/dinov2
HouseMaker uses only feed-forward symmetry detection, not diffusion generation.
"""
_setup_lock = threading.Lock()
_validated_assets: dict[Path, tuple[int, int, str]] = {}


# ### Public setup data, errors, and paths ###
class Reflect3DError(RuntimeError):
    """An explicit local symmetry setup or inference failure."""


class Reflect3DCancelled(Reflect3DError):
    """The caller interrupted symmetry inference or its optional setup."""


@dataclass(frozen=True)
class Reflect3DBundle:
    directory: Path
    python: Path
    dependencies: Path

    @property
    def source(self) -> Path:
        return self.directory / "source" / "feedforward_symm"

    @property
    def checkpoint(self) -> Path:
        return self.directory / REFLECT3D_WEIGHTS_FILENAME


def default_reflect3d_directory() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    return base / "HouseMaker" / "models" / "reflect3d"


def check_reflect3d_cancelled(cancellation_check=None) -> None:
    if cancellation_check is not None and cancellation_check():
        raise Reflect3DCancelled("Local Reflect3D symmetry inference was cancelled.")


# ### Cancellable, checksum-verified downloads ###
def _sha256(path: Path, cancellation_check=None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            check_reflect3d_cancelled(cancellation_check)
            digest.update(chunk)
    return digest.hexdigest()


def _verified_asset(path: Path, expected: str, cancellation_check=None) -> bool:
    if not path.is_file():
        return False
    stat = path.stat()
    signature = (stat.st_size, stat.st_mtime_ns, expected)
    if _validated_assets.get(path) != signature:
        if _sha256(path, cancellation_check) != expected:
            raise Reflect3DError(f"Reflect3D asset checksum mismatch: {path}. Remove it and retry setup.")
        _validated_assets[path] = signature
    return True


def _download_asset(url: str, destination: Path, expected: str, *,
                    cancellation_check=None, progress_callback=None) -> None:
    if _verified_asset(destination, expected, cancellation_check):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    try:
        request = Request(url, headers={"User-Agent": "HouseMaker-Reflect3D/1.0"})
        digest, count, last_report = hashlib.sha256(), 0, 0.0
        with urlopen(request, timeout=30) as response, temporary.open("wb") as output:
            if "text/html" in response.headers.get("Content-Type", ""):
                raise Reflect3DError(f"The Reflect3D server returned a webpage instead of {destination.name}.")
            total = int(response.headers.get("Content-Length") or 0)
            while chunk := response.read(256 * 1024):
                check_reflect3d_cancelled(cancellation_check)
                output.write(chunk)
                digest.update(chunk)
                count += len(chunk)
                if progress_callback and time.monotonic() - last_report >= 1:
                    suffix = f" / {total // (1024 * 1024)} MB" if total else " MB"
                    progress_callback(f"Downloading Reflect3D {destination.name}: {count // (1024 * 1024)}{suffix}")
                    last_report = time.monotonic()
        if digest.hexdigest() != expected:
            raise Reflect3DError(f"The downloaded Reflect3D {destination.name} failed its SHA-256 check.")
        check_reflect3d_cancelled(cancellation_check)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


# ### Isolated setup subprocesses ###
def stop_reflect3d_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW, check=False,
        )
    else:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _run_setup(command: list[str], *, cancellation_check=None, progress_callback=None) -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.update({"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", env=environment,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    lines: queue.Queue[str] = queue.Queue()

    def read_output() -> None:
        if process.stdout is not None:
            for line in process.stdout:
                lines.put(line.strip())

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    tail: list[str] = []
    last_report = 0.0
    try:
        while process.poll() is None or reader.is_alive() or not lines.empty():
            check_reflect3d_cancelled(cancellation_check)
            try:
                line = lines.get(timeout=0.1)
                if line:
                    tail = [*tail[-24:], line]
            except queue.Empty:
                pass
            if progress_callback and time.monotonic() - last_report >= 2:
                progress_callback(f"Preparing Reflect3D runtime: {tail[-1][:180]}" if tail else "Preparing Reflect3D runtime")
                last_report = time.monotonic()
        if process.returncode:
            raise Reflect3DError("Reflect3D runtime setup failed: " + "\n".join(tail[-8:]))
    finally:
        stop_reflect3d_process(process)
        reader.join(timeout=2)
        if process.stdout is not None:
            process.stdout.close()


def _runtime_python(directory: Path) -> Path:
    return directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _has_compatible_torch(python: Path) -> bool:
    if not python.is_file():
        return False
    # Metadata inspection does not import Torch or initialize CUDA.
    check = subprocess.run(
        [str(python), "-c", "import importlib.metadata as m; print(m.version('torch'))"],
        capture_output=True, text=True, timeout=20, check=False,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    return check.returncode == 0 and check.stdout.strip().split("+")[0] == REFLECT3D_TORCH_VERSION


def _prepare_runtime(directory: Path, *, cancellation_check=None, progress_callback=None) -> tuple[Path, Path]:
    runtime = directory / "runtime"
    python = _runtime_python(runtime)
    # Reuse an existing optional CUDA runtime read-only. Extra dependencies go
    # ONLY into Reflect3D's target directory; never pip-install into CubiCasa.
    existing = _runtime_python(directory.parent / "cubicasa5k" / "runtime")
    if not _has_compatible_torch(python):
        if _has_compatible_torch(existing):
            python = existing
        else:
            check_reflect3d_cancelled(cancellation_check)
            if not python.is_file():
                venv.EnvBuilder(with_pip=True).create(runtime)
            _run_setup(
                [str(python), "-m", "pip", "install", f"torch=={REFLECT3D_TORCH_VERSION}",
                 "--index-url", "https://download.pytorch.org/whl/cu124"],
                cancellation_check=cancellation_check, progress_callback=progress_callback,
            )
    dependencies = directory / "dependencies"
    marker = dependencies / "reflect3d-packages.json"
    expected = {"version": REFLECT3D_VERSION, "packages": list(REFLECT3D_RUNTIME_PACKAGES)}
    try:
        ready = json.loads(marker.read_text(encoding="utf-8")) == expected
    except (OSError, ValueError):
        ready = False
    if not ready:
        dependencies.mkdir(parents=True, exist_ok=True)
        _run_setup(
            [str(python), "-m", "pip", "install", "--upgrade", "--target", str(dependencies),
             *REFLECT3D_RUNTIME_PACKAGES],
            cancellation_check=cancellation_check, progress_callback=progress_callback,
        )
        check_reflect3d_cancelled(cancellation_check)
        marker.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    return python, dependencies


# ### Public lazy setup entry point ###
def ensure_reflect3d_model(*, cancellation_check=None, progress_callback=None) -> Reflect3DBundle:
    while not _setup_lock.acquire(timeout=0.1):
        check_reflect3d_cancelled(cancellation_check)
    try:
        check_reflect3d_cancelled(cancellation_check)
        directory = default_reflect3d_directory()
        directory.mkdir(parents=True, exist_ok=True)
        if progress_callback:
            progress_callback("Preparing local Reflect3D symmetry detector (upstream license unspecified).")
        for relative, checksum in REFLECT3D_SOURCE_SHA256.items():
            _download_asset(
                REFLECT3D_SOURCE_ROOT + relative, directory / "source" / "feedforward_symm" / relative,
                checksum, cancellation_check=cancellation_check, progress_callback=progress_callback,
            )
        _download_asset(
            REFLECT3D_WEIGHTS_URL, directory / REFLECT3D_WEIGHTS_FILENAME,
            REFLECT3D_WEIGHTS_SHA256, cancellation_check=cancellation_check, progress_callback=progress_callback,
        )
        python, dependencies = _prepare_runtime(
            directory, cancellation_check=cancellation_check, progress_callback=progress_callback,
        )
        from .reflect3d_reference_symmetry import reflect3d_policy_signature

        (directory / "NOTICE.txt").write_text(REFLECT3D_NOTICE, encoding="utf-8")
        (directory / "reflect3d-manifest.json").write_text(json.dumps({
            "model": REFLECT3D_MODEL_ID, "version": REFLECT3D_VERSION,
            "revision": REFLECT3D_REVISION, "weights_sha256": REFLECT3D_WEIGHTS_SHA256,
            "source_sha256": REFLECT3D_SOURCE_SHA256, "license": "unspecified by upstream",
            "python": str(python), "packages": list(REFLECT3D_RUNTIME_PACKAGES),
            "acceptance_policy": reflect3d_policy_signature(),
        }, indent=2), encoding="utf-8")
        return Reflect3DBundle(directory, python, dependencies)
    except Reflect3DError:
        raise
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise Reflect3DError(f"The local Reflect3D detector could not be prepared: {exc}") from exc
    finally:
        _setup_lock.release()


# ### Optional explicit setup command ###
if __name__ == "__main__":
    try:
        installed = ensure_reflect3d_model(progress_callback=print)
        print(f"Reflect3D prepared: {installed.directory}")
    except Reflect3DError as error:
        print(f"Reflect3D setup failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
