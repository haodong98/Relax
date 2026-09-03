#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# One-time Relax training environment bootstrap, run *inside* the EDF
# container (Python 3.12, torch, sglang -- no Megatron-LM,
# transformer_engine, or apex). Idempotent: skips everything if
# the completion marker already exists on persistent storage, so repeated
# job submissions after the first only cost a few seconds.
#
# apex is deliberately avoided: it is not imported by relax/backends/megatron/
# directly (--transformer-impl local is used instead).
#
# transformer_engine, however, is NOT optional under --megatron-to-hf-mode
# bridge, even with --transformer-impl local: megatron/bridge/peft/lora_layers.py
# does a module-level `import transformer_engine.pytorch`, which the
# `import megatron.bridge` chain always reaches. Same for nvidia-modelopt
# (megatron/bridge/models/conversion/auto_bridge.py) and torchaudio -- the
# container ships a torchaudio whose .so is ABI-incompatible with its own
# torch, and transformers' audio_utils imports it unconditionally, so a
# matching build has to shadow it from the venv.

set -euo pipefail

RELAX_ENV_ROOT="${1:?usage: setup_relax_env.sh <persistent-root-dir> <relax-repo-dir> [mobilegym-repo-dir]}"
RELAX_REPO_DIR="${2:?usage: setup_relax_env.sh <persistent-root-dir> <relax-repo-dir> [mobilegym-repo-dir]}"
MOBILEGYM_REPO_DIR="${3:-}"

VENV_DIR="${RELAX_ENV_ROOT}/relax_venv"
MEGATRON_DIR="${RELAX_ENV_ROOT}/Megatron-LM"
MEGATRON_BRIDGE_DIR="${RELAX_ENV_ROOT}/Megatron-Bridge"
# Keep in sync with docker/Dockerfile's MEGATRON_BRIDGE_COMMIT.
MEGATRON_BRIDGE_COMMIT="${MEGATRON_BRIDGE_COMMIT:-2faedbf6fe3c422835a44b2b360cadcb2a116a54}"
MODELOPT_VERSION="${MODELOPT_VERSION:-0.44.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.9.1}"   # must match the container's torch
TE_VERSION="${TE_VERSION:-2.14.1}"
TE_SOURCE_COMMIT="${TE_SOURCE_COMMIT:-366798ef8a0a00d8f2c1650d11e7e623d7c33e26}"
TE_CUDA_ARCHS="${TE_CUDA_ARCHS:-90}"
CUDNN_VERSION="${CUDNN_VERSION:-9.16.0.29}"
CUBLAS_VERSION="${CUBLAS_VERSION:-13.0.0.19}"
CUDA_NVRTC_VERSION="${CUDA_NVRTC_VERSION:-13.0.48}"
ONNXSCRIPT_VERSION="${ONNXSCRIPT_VERSION:-0.7.1}"
NUMPY_VERSION="${NUMPY_VERSION:-1.26.4}"
TE_SOURCE_DIR="${RELAX_ENV_ROOT}/TransformerEngine-${TE_VERSION}"
TRANSFER_QUEUE_DIR="${RELAX_ENV_ROOT}/TransferQueue"
MARKER="${RELAX_ENV_ROOT}/.setup_complete"
PIP="${VENV_DIR}/bin/pip"

# MobileGym runs under this container-built Python, not the host-side G1 venv.
# Keep its already-declared runtime dependencies as a separately versioned,
# additive layer so an existing expensive TE/Bridge bootstrap can be upgraded
# without silently rebuilding or mutating those compiled components.
MOBILEGYM_REQUIREMENTS_SHA256="ce27ba090564035fb1c1cb3d6c3c8691270487f1301fd17b5b1a465562bec1e3"
MOBILEGYM_DEPS_MARKER="${RELAX_ENV_ROOT}/.mobilegym_runtime_py312_v2"

install_mobilegym_runtime() {
    if [ -z "${MOBILEGYM_REPO_DIR}" ]; then
        return
    fi
    local requirements_file="${MOBILEGYM_REPO_DIR}/bench_env/requirements.txt"
    if [ ! -r "${requirements_file}" ]; then
        echo "[setup_relax_env] ERROR: MobileGym requirements missing: ${requirements_file}" >&2
        exit 1
    fi
    local actual_requirements_sha256
    actual_requirements_sha256="$(sha256sum "${requirements_file}" | awk '{print $1}')"
    if [ "${actual_requirements_sha256}" != "${MOBILEGYM_REQUIREMENTS_SHA256}" ]; then
        echo "[setup_relax_env] ERROR: MobileGym requirements changed (${actual_requirements_sha256})." >&2
        echo "[setup_relax_env] Review and version the runtime pins before using this checkout." >&2
        exit 1
    fi
    if [ ! -f "${MOBILEGYM_DEPS_MARKER}" ]; then
        echo "[setup_relax_env] Installing pinned MobileGym runtime dependencies"
        "${PIP}" install -q \
            "openai==2.6.1" \
            "playwright==1.62.0" \
            "pillow==11.3.0" \
            "tqdm==4.70.0" \
            "opencc-python-reimplemented==0.1.7"
    fi
    PYTHONPATH="${MOBILEGYM_REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}" "${VENV_DIR}/bin/python" - <<'PY'
from importlib.metadata import version
import bench_env  # noqa: F401
import playwright  # noqa: F401

expected = {
    "openai": "2.6.1",
    "playwright": "1.62.0",
    "pillow": "11.3.0",
    "tqdm": "4.70.0",
    "opencc-python-reimplemented": "0.1.7",
}
for package, expected_version in expected.items():
    assert version(package) == expected_version, (package, version(package), expected_version)
PY
    touch "${MOBILEGYM_DEPS_MARKER}"
}

verify_cuda13_package_set() {
    EXPECTED_CUDNN_VERSION="${CUDNN_VERSION}" EXPECTED_CUBLAS_VERSION="${CUBLAS_VERSION}" \
        EXPECTED_CUDA_NVRTC_VERSION="${CUDA_NVRTC_VERSION}" EXPECTED_CUDNN_ROOT="${CUDNN_ROOT}" \
        "${VENV_DIR}/bin/python" - <<'PY'
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import re

expected = {
    "nvidia-cudnn-cu13": os.environ["EXPECTED_CUDNN_VERSION"],
    "nvidia-cublas": os.environ["EXPECTED_CUBLAS_VERSION"],
    "nvidia-cuda-nvrtc": os.environ["EXPECTED_CUDA_NVRTC_VERSION"],
}
for package, expected_version in expected.items():
    assert version(package) == expected_version, (package, version(package), expected_version)
for package in ("nvidia-cudnn-cu12", "nvidia-cublas-cu12", "nvidia-cuda-nvrtc-cu12"):
    try:
        installed_version = version(package)
    except PackageNotFoundError:
        continue
    raise AssertionError(f"CUDA 12 package must not be present: {package}=={installed_version}")

header = Path(os.environ["EXPECTED_CUDNN_ROOT"]) / "include" / "cudnn_version.h"
header_text = header.read_text(encoding="utf-8")
actual_header_version = tuple(
    int(re.search(rf"#define CUDNN_{name} (\d+)", header_text).group(1))
    for name in ("MAJOR", "MINOR", "PATCHLEVEL")
)
expected_header_version = tuple(int(part) for part in os.environ["EXPECTED_CUDNN_VERSION"].split(".")[:3])
assert actual_header_version == expected_header_version, (header, actual_header_version, expected_header_version)
PY
}

run_te_fused_attention_smoke() {
    echo "[setup_relax_env] Running real Transformer Engine FusedAttention forward/backward smoke"
    local cudnn_preload
    local cudnn_library
    local -a cudnn_libraries=(
        libcudnn.so.9
        libcudnn_graph.so.9
        libcudnn_engines_runtime_compiled.so.9
        libcudnn_ops.so.9
        libcudnn_cnn.so.9
        libcudnn_adv.so.9
        libcudnn_engines_precompiled.so.9
        libcudnn_heuristic.so.9
    )
    local -a cudnn_preload_paths=()
    for cudnn_library in "${cudnn_libraries[@]}"; do
        if [ ! -r "${CUDNN_LIB_DIR}/${cudnn_library}" ]; then
            echo "[setup_relax_env] ERROR: missing cuDNN preload library: ${CUDNN_LIB_DIR}/${cudnn_library}." >&2
            exit 1
        fi
        cudnn_preload_paths+=("${CUDNN_LIB_DIR}/${cudnn_library}")
    done
    cudnn_preload="$(IFS=:; echo "${cudnn_preload_paths[*]}")"
    if ! EXPECTED_CUDNN_LIB_DIR="${CUDNN_LIB_DIR}" EXPECTED_CUDNN_VERSION="${CUDNN_VERSION}" \
        EXPECTED_VENV_DIR="${VENV_DIR}" \
        CUDNN_LOGERR_DBG=1 CUDNN_LOGDEST_DBG=stderr \
        NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=1 NVTE_UNFUSED_ATTN=0 \
        LD_PRELOAD="${cudnn_preload}${LD_PRELOAD:+:${LD_PRELOAD}}" \
        LD_LIBRARY_PATH="${CUDNN_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
        PYTHONPATH="${RELAX_REPO_DIR}:${MEGATRON_DIR}:${TRANSFER_QUEUE_DIR}:${VENV_SITE}" python3.12 - <<'PY'
import ctypes
import json
import os
from importlib.metadata import distribution, version
from pathlib import Path

import torch
from transformer_engine.pytorch.attention import DotProductAttention
from transformer_engine.pytorch.attention.dot_product_attention import _attention_backends
from transformer_engine.pytorch.cpp_extensions import NVTE_Fused_Attn_Backend
from torch.utils.cpp_extension import CUDA_HOME

assert torch.cuda.is_available(), "CUDA is unavailable during the FusedAttention runtime gate"
expected_cudnn_dir = str(Path(os.environ["EXPECTED_CUDNN_LIB_DIR"]).resolve())
expected_cudnn_parts = tuple(int(part) for part in os.environ["EXPECTED_CUDNN_VERSION"].split(".")[:3])
expected_cudnn_runtime = expected_cudnn_parts[0] * 10000 + expected_cudnn_parts[1] * 100 + expected_cudnn_parts[2]
assert torch.backends.cudnn.version() == expected_cudnn_runtime, (
    torch.backends.cudnn.version(),
    expected_cudnn_runtime,
)
global_cudnn_get_version = ctypes.CDLL(None).cudnnGetVersion
global_cudnn_get_version.restype = ctypes.c_size_t
assert global_cudnn_get_version() == expected_cudnn_runtime, (
    global_cudnn_get_version(),
    expected_cudnn_runtime,
)


def mapped_library_paths() -> list[str]:
    return sorted(
        {
            fields[-1]
            for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
            if len(fields := line.split()) >= 6 and fields[-1].startswith("/")
        }
    )


pre_forward_cudnn_paths = [
    path for path in mapped_library_paths() if Path(path).name.startswith("libcudnn")
]
assert pre_forward_cudnn_paths, "LD_PRELOAD did not map any cuDNN library"
foreign_cudnn_paths = [
    path
    for path in pre_forward_cudnn_paths
    if not str(Path(path).resolve()).startswith(expected_cudnn_dir + os.sep)
]
# The EDF runtime maps a second, versioned main dispatch DSO from /usr/lib even
# when the selected cuDNN is preloaded. It is acceptable only while both the
# global symbol and Torch resolve to 9.16; foreign component libraries would
# create a mixed execution stack and remain a hard failure.
assert all(Path(path).name.startswith("libcudnn.so.9") for path in foreign_cudnn_paths), foreign_cudnn_paths

torch.manual_seed(17)
attention = DotProductAttention(8, 64, attention_dropout=0.0, attn_mask_type="no_mask").to(
    device="cuda", dtype=torch.bfloat16
)
q, k, v = (
    torch.randn(1024, 2, 8, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    for _ in range(3)
)
output = attention(q, k, v)
loss = output.float().square().mean()
loss.backward()
torch.cuda.synchronize()
assert torch.isfinite(output).all() and torch.isfinite(loss), (output, loss)
assert all(
    tensor.grad is not None and torch.isfinite(tensor.grad).all() and torch.count_nonzero(tensor.grad) > 0
    for tensor in (q, k, v)
)
assert bool(_attention_backends["use_fused_attention"]), _attention_backends
assert not bool(_attention_backends["use_flash_attention"]), _attention_backends
assert not bool(_attention_backends["use_unfused_attention"]), _attention_backends
expected_backend = NVTE_Fused_Attn_Backend.NVTE_F16_arbitrary_seqlen
assert _attention_backends["fused_attention_backend"] == expected_backend, _attention_backends

mapped_paths = mapped_library_paths()
cudnn_paths = [path for path in mapped_paths if Path(path).name.startswith("libcudnn")]
assert cudnn_paths, "FusedAttention did not map any cuDNN library"
post_forward_foreign_cudnn_paths = [
    path for path in cudnn_paths if not str(Path(path).resolve()).startswith(expected_cudnn_dir + os.sep)
]
assert all(
    Path(path).name.startswith("libcudnn.so.9") for path in post_forward_foreign_cudnn_paths
), post_forward_foreign_cudnn_paths
legacy_cuda_paths = [
    path for path in mapped_paths if Path(path).name.startswith(("libcublas", "libcud", "libnvrtc")) and ".so.12" in Path(path).name
]
assert not legacy_cuda_paths, legacy_cuda_paths
cuda13_lib_dir = Path(distribution("nvidia-cublas").locate_file("nvidia/cu13/lib")).resolve()
assert not str(cuda13_lib_dir).startswith(str(Path(os.environ["EXPECTED_VENV_DIR"]).resolve()) + os.sep)
cublas_paths = [path for path in mapped_paths if Path(path).name.startswith("libcublas")]
nvrtc_paths = [path for path in mapped_paths if Path(path).name.startswith("libnvrtc")]
cudart_paths = [path for path in mapped_paths if Path(path).name.startswith("libcudart")]
assert all(
    ".so.13" in Path(path).name and str(Path(path).resolve()).startswith(str(cuda13_lib_dir) + os.sep)
    for path in cublas_paths
), cublas_paths
assert CUDA_HOME is not None, "Torch did not resolve the frozen CUDA toolkit root"
cuda_toolkit_dir = Path(CUDA_HOME).resolve()
assert all(
    ".so.13" in Path(path).name
    and any(
        str(Path(path).resolve()).startswith(str(root) + os.sep)
        for root in (cuda13_lib_dir, cuda_toolkit_dir)
    )
    for path in nvrtc_paths
), nvrtc_paths
assert all(".so.13" in Path(path).name for path in cudart_paths), cudart_paths
print(
    json.dumps(
        {
            "backend": str(expected_backend),
            "cublas_package_version": version("nvidia-cublas"),
            "cuda_nvrtc_package_version": version("nvidia-cuda-nvrtc"),
            "cudnn_runtime_version": torch.backends.cudnn.version(),
            "cudnn_paths": cudnn_paths,
            "foreign_cudnn_main_paths": post_forward_foreign_cudnn_paths,
            "cublas_paths": cublas_paths,
            "nvrtc_paths": nvrtc_paths,
            "cuda_toolkit_dir": str(cuda_toolkit_dir),
            "cudart_paths": cudart_paths,
            "device": torch.cuda.get_device_name(0),
            "torch_cuda_version": torch.version.cuda,
            "torch_version": torch.__version__,
            "transformer_engine_version": version("transformer-engine"),
        },
        sort_keys=True,
    )
)
PY
    then
        echo "[setup_relax_env] ERROR: real FusedAttention runtime gate failed." >&2
        exit 1
    fi
}

if [ -f "${MARKER}" ]; then
    echo "[setup_relax_env] ${MARKER} exists; validating CUDA 13 runtime and additive layers."
    VENV_SITE="$("${VENV_DIR}/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")"
    CUDNN_ROOT="${VENV_SITE}/nvidia/cudnn"
    CUDNN_LIB_DIR="${CUDNN_ROOT}/lib"
    install_mobilegym_runtime
    verify_cuda13_package_set
    run_te_fused_attention_smoke
    exit 0
fi

echo "[setup_relax_env] Creating venv at ${VENV_DIR} (--system-site-packages inherits the container's torch/sglang)"
python3.12 -m venv --system-site-packages "${VENV_DIR}"
"${PIP}" install -q -U pip

echo "[setup_relax_env] Installing Relax requirements + editable package"
"${PIP}" install -q -r "${RELAX_REPO_DIR}/requirements.txt"
"${PIP}" install -q -e "${RELAX_REPO_DIR}" --no-deps

# Bridge-mode dependencies the container does not supply -- see header. Pinned
# to the versions validated against this image (torch 2.9.1+cu130).
echo "[setup_relax_env] Installing bridge-mode deps (modelopt, torchaudio, cuDNN, ONNX, build tools)"
"${PIP}" install -q "nvidia-modelopt==${MODELOPT_VERSION}"
# --no-deps: torchaudio is here only to shadow the container's broken build,
# and its dependency metadata would otherwise drag torch in behind it.
"${PIP}" install -q --no-deps --index-url https://download.pytorch.org/whl/cu130 \
    "torchaudio==${TORCHAUDIO_VERSION}"
# onnxscript's unconstrained dependency would otherwise upgrade NumPy to 2.x,
# while Relax and the container's binary pyarrow stack require the NumPy 1.x
# ABI. Resolve both constraints in one pip transaction.
"${PIP}" install -q "numpy==${NUMPY_VERSION}" "onnxscript==${ONNXSCRIPT_VERSION}" pybind11 ninja
# Torch in the frozen container already provides its exact CUDA 13 cuBLAS and
# NVRTC packages. Install only the cuDNN override: pulling its dependency into
# the venv would duplicate and potentially upgrade Torch's CUDA runtime.
"${PIP}" install -q --no-deps "nvidia-cudnn-cu13==${CUDNN_VERSION}"
VENV_SITE="$("${VENV_DIR}/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")"
CUDNN_ROOT="${VENV_SITE}/nvidia/cudnn"
CUDNN_LIB_DIR="${CUDNN_ROOT}/lib"
if [ ! -r "${CUDNN_ROOT}/include/cudnn.h" ] || [ ! -d "${CUDNN_LIB_DIR}" ]; then
    echo "[setup_relax_env] ERROR: cuDNN ${CUDNN_VERSION} is incomplete under ${CUDNN_ROOT}." >&2
    exit 1
fi
verify_cuda13_package_set

# CUDA 13's published Transformer Engine wheels are not ABI-compatible with
# this container's torch. Build the fixed upstream source revision in the
# container instead. GH200 is SM90, so do not spend bootstrap time compiling
# irrelevant architectures.
if [ ! -d "${TE_SOURCE_DIR}/.git" ]; then
    echo "[setup_relax_env] Cloning Transformer Engine @ ${TE_SOURCE_COMMIT}"
    git clone --depth 1 --branch "v${TE_VERSION}" --recurse-submodules \
        https://github.com/NVIDIA/TransformerEngine.git "${TE_SOURCE_DIR}"
fi
if [ "$(git -C "${TE_SOURCE_DIR}" rev-parse HEAD)" != "${TE_SOURCE_COMMIT}" ]; then
    echo "[setup_relax_env] ERROR: ${TE_SOURCE_DIR} is not pinned to ${TE_SOURCE_COMMIT}." >&2
    echo "[setup_relax_env] Use a new versioned RELAX_ENV_ROOT rather than mixing builds." >&2
    exit 1
fi
echo "[setup_relax_env] Building Transformer Engine ${TE_VERSION} for SM${TE_CUDA_ARCHS}"
CMAKE_PREFIX_PATH="${CUDNN_ROOT}" LD_LIBRARY_PATH="${CUDNN_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    NVTE_CUDA_ARCHS="${TE_CUDA_ARCHS}" "${PIP}" install -q --no-build-isolation --no-deps "${TE_SOURCE_DIR}"

# relax/backends/megatron/model_provider.py imports `megatron.bridge` (and so
# does relax/models), so a plain upstream Megatron-LM clone is not enough: the
# tree has to be the Megatron-Bridge merge that docker/Dockerfile builds --
# megatron/bridge from Bridge's src, over the mcore revision Bridge pins in
# 3rdparty. Upstream main is *not* that revision, so megatron/ is replaced
# wholesale (empty dir first) rather than overlaid, which would otherwise leave
# stale modules from the newer upstream tree behind.
if [ ! -d "${MEGATRON_DIR}/megatron/bridge" ]; then
    echo "[setup_relax_env] Cloning Megatron-Bridge @ ${MEGATRON_BRIDGE_COMMIT}"
    rm -rf "${MEGATRON_BRIDGE_DIR}" "${MEGATRON_DIR}"
    git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git "${MEGATRON_BRIDGE_DIR}"
    git -C "${MEGATRON_BRIDGE_DIR}" checkout "${MEGATRON_BRIDGE_COMMIT}"
    git -C "${MEGATRON_BRIDGE_DIR}" submodule update --init --recursive
    (cd "${MEGATRON_BRIDGE_DIR}" && ./scripts/switch_mcore.sh dev)

    echo "[setup_relax_env] Assembling ${MEGATRON_DIR} from Megatron-Bridge"
    mkdir -p "${MEGATRON_DIR}"
    # rsync is not in the EDF image.  `cp -a` normally preserves mode,
    # ownership, SELinux context, and xattrs, but the EDF container's view of
    # /iopsstor rejects those metadata operations with EINVAL.  Keep the
    # recursive/link/timestamp semantics while omitting only unsupported
    # metadata; code content and symlink layout remain unchanged.
    cp -a --no-preserve=mode,ownership,context,xattr "${MEGATRON_BRIDGE_DIR}/src/megatron" "${MEGATRON_DIR}/"
    cp -a --no-preserve=mode,ownership,context,xattr "${MEGATRON_BRIDGE_DIR}/3rdparty/Megatron-LM/megatron/." "${MEGATRON_DIR}/megatron/"
fi

# relax/core/controller.py imports transfer_queue unconditionally, but it is
# an internal redai-infra package (not on PyPI, not vendored in this
# checkout, not in requirements.txt -- see AGENTS.md's file listing vs. what
# is actually checked out, and .github/workflows/ci.yml's CI-only stub for
# the same reason). redai-infra/TransferQueue (not the unrelated
# Ascend/TransferQueue also cited in README.md's references section) is the
# right source: same GitHub org as Relax itself (setup.py), with feature
# branches (feat/rednote-ai/...) that track this codebase.
if [ ! -d "${TRANSFER_QUEUE_DIR}/.git" ]; then
    echo "[setup_relax_env] Cloning TransferQueue to ${TRANSFER_QUEUE_DIR}"
    git clone --depth 1 https://github.com/redai-infra/TransferQueue.git "${TRANSFER_QUEUE_DIR}"
fi
"${PIP}" install -q -e "${TRANSFER_QUEUE_DIR}"

# TransferQueue installs tensordict, whose dependency resolution follows the
# container Torch metadata and can restore Torch's original cuDNN package.
# Apply the validated override after every Torch-adjacent dependency install,
# then validate the real header as well as package metadata.
"${PIP}" install -q --no-deps --force-reinstall "nvidia-cudnn-cu13==${CUDNN_VERSION}"
verify_cuda13_package_set

# Fail here rather than 20 minutes into a multi-node run. Transformer Engine's
# extension must load through the venv's cuDNN library in both the venv and the
# container Python used by Ray workers.
echo "[setup_relax_env] Verifying bridge-mode imports"
if ! LD_LIBRARY_PATH="${CUDNN_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    PYTHONPATH="${MEGATRON_DIR}" "${VENV_DIR}/bin/python" -c "
from importlib.metadata import version
from megatron.core.extensions.transformer_engine import HAVE_TE
assert HAVE_TE, 'transformer_engine imported but its torch extension did not load'
from megatron.bridge import AutoBridge  # noqa: F401
import torchaudio
assert torchaudio.__version__.split('+', 1)[0] == '${TORCHAUDIO_VERSION}'
assert str(torchaudio.__file__).startswith('${VENV_DIR}/')
assert version('transformer-engine').split('+', 1)[0] == '${TE_VERSION}'
"; then
    echo "[setup_relax_env] ERROR: bridge-mode imports are broken; not writing ${MARKER}." >&2
    echo "[setup_relax_env] Transformer Engine must be rebuilt in this container; do not reuse" >&2
    echo "[setup_relax_env] a root whose source revision or cuDNN runtime differs." >&2
    echo "[setup_relax_env] If it is \"'qwen3_asr' is already used by a Transformers config\"," >&2
    echo "[setup_relax_env] transformers is newer than this Megatron-Bridge commit: Bridge still" >&2
    echo "[setup_relax_env] registers qwen3_asr itself, which transformers >=5.15 ships natively." >&2
    echo "[setup_relax_env] requirements.txt leaves transformers unpinned -- hold it at the" >&2
    echo "[setup_relax_env] container's version (5.3.0 works) or move MEGATRON_BRIDGE_COMMIT up." >&2
    exit 1
fi

# Ray workers run under the container interpreter, not VENV_DIR/bin/python.
# The launcher exposes the venv through PYTHONPATH, so prove that exact import
# contract now rather than discovering a missing bridge/TE package after Ray
# has allocated a multi-node training job.
echo "[setup_relax_env] Verifying container-Python runtime imports"
if ! EXPECTED_TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION}" EXPECTED_VENV_DIR="${VENV_DIR}" \
    LD_LIBRARY_PATH="${CUDNN_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    PYTHONPATH="${RELAX_REPO_DIR}:${MEGATRON_DIR}:${TRANSFER_QUEUE_DIR}:${VENV_SITE}" python3.12 - <<'PY'
import os

from megatron.bridge import AutoBridge  # noqa: F401
from megatron.core.extensions.transformer_engine import HAVE_TE

assert HAVE_TE, "Transformer Engine extension unavailable to container Python"
import relax  # noqa: F401
import torchaudio
import transfer_queue  # noqa: F401

assert torchaudio.__version__.split("+", 1)[0] == os.environ["EXPECTED_TORCHAUDIO_VERSION"]
assert str(torchaudio.__file__).startswith(os.environ["EXPECTED_VENV_DIR"] + "/")
PY
then
    echo "[setup_relax_env] ERROR: container Python cannot import the Bridge runtime." >&2
    exit 1
fi

install_mobilegym_runtime
verify_cuda13_package_set
run_te_fused_attention_smoke
touch "${MARKER}"
echo "[setup_relax_env] Done."
