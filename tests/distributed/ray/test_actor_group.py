# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from pathlib import Path

import pytest

from relax.distributed.ray.actor_group import _CUDNN_PRELOAD_LIBRARIES, _ensure_cudnn_library_precedence


def _create_cudnn_libraries(cudnn_dir: Path) -> list[Path]:
    cudnn_dir.mkdir(parents=True)
    libraries = [cudnn_dir / name for name in _CUDNN_PRELOAD_LIBRARIES]
    for library in libraries:
        library.touch()
    return libraries


def test_ensure_cudnn_library_precedence_deduplicates_and_prepends(tmp_path: Path) -> None:
    cudnn_dir = tmp_path / "cudnn" / "lib"
    cudnn_libraries = _create_cudnn_libraries(cudnn_dir)
    env_vars = {
        "CUDNN_LIB_DIR": str(cudnn_dir),
        "LD_LIBRARY_PATH": f"/opencv/lib:{cudnn_dir}:/cuda/lib:{cudnn_dir}",
        "LD_PRELOAD": "/existing/hook.so",
    }

    _ensure_cudnn_library_precedence(env_vars)

    assert env_vars["LD_LIBRARY_PATH"].split(":") == [str(cudnn_dir), "/opencv/lib", "/cuda/lib"]
    assert env_vars["LD_PRELOAD"].split(":") == [str(path) for path in cudnn_libraries] + ["/existing/hook.so"]


def test_ensure_cudnn_library_precedence_rejects_missing_directory(tmp_path: Path) -> None:
    env_vars = {"CUDNN_LIB_DIR": str(tmp_path / "missing"), "LD_LIBRARY_PATH": "/cuda/lib"}

    with pytest.raises(RuntimeError, match="CUDNN_LIB_DIR does not exist for train actors"):
        _ensure_cudnn_library_precedence(env_vars)


def test_ensure_cudnn_library_precedence_requires_all_runtime_libraries(tmp_path: Path) -> None:
    cudnn_dir = tmp_path / "cudnn" / "lib"
    cudnn_dir.mkdir(parents=True)
    (cudnn_dir / "libcudnn.so.9").touch()

    with pytest.raises(RuntimeError, match="missing required train-actor libraries"):
        _ensure_cudnn_library_precedence({"CUDNN_LIB_DIR": str(cudnn_dir)})


def test_ensure_cudnn_library_precedence_rejects_conflicting_preload(tmp_path: Path) -> None:
    cudnn_dir = tmp_path / "cudnn" / "lib"
    _create_cudnn_libraries(cudnn_dir)
    foreign_cudnn = tmp_path / "foreign" / "libcudnn.so.9"
    foreign_cudnn.parent.mkdir()
    foreign_cudnn.touch()

    with pytest.raises(RuntimeError, match="conflicting cuDNN library"):
        _ensure_cudnn_library_precedence(
            {"CUDNN_LIB_DIR": str(cudnn_dir), "LD_PRELOAD": f"/existing/hook.so:{foreign_cudnn}"}
        )
