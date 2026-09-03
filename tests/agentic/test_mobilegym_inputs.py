# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


def test_mobilegym_task_manifest_is_byte_stable_and_has_sample_seed(tmp_path: Path) -> None:
    mobilegym_repo = tmp_path / "mobilegym"
    split = mobilegym_repo / "bench_env" / "splits"
    split.mkdir(parents=True)
    (split / "train.txt").write_text("task.alpha\ntask.beta\n", encoding="utf-8")
    script = Path("examples/mobilegym_agentic/scripts/build_tasks_jsonl.py")
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    base = [sys.executable, str(script), "--mobilegym-repo", str(mobilegym_repo), "--repeat", "2"]

    subprocess.run([*base, "--output", str(first)], check=True, capture_output=True, text=True)
    subprocess.run([*base, "--output", str(second)], check=True, capture_output=True, text=True)

    assert first.read_bytes() == second.read_bytes()
    rows = [json.loads(line) for line in first.read_text(encoding="utf-8").splitlines()]
    assert [row["metadata"]["task_id"] for row in rows] == ["task.alpha", "task.alpha", "task.beta", "task.beta"]
    assert all(isinstance(row["metadata"]["sample_seed"], int) for row in rows)
    assert all(0 <= row["metadata"]["sample_seed"] <= 2**32 - 1 for row in rows)
    assert rows[0]["metadata"]["sample_seed"] != rows[1]["metadata"]["sample_seed"]


def test_direct_experiment_config_generator_emits_only_two_dual_arms(tmp_path: Path) -> None:
    output_dir = tmp_path / "configs"
    subprocess.run(
        [
            sys.executable,
            "examples/agentic_dual_judge/prepare_benchmark_configs.py",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    generated = sorted(output_dir.glob("*.json"))
    assert [path.name for path in generated] == [
        "judge_services_dual_per_turn.json",
        "judge_services_dual_terminal_once.json",
    ]
    assert {
        (
            json.loads(path.read_text(encoding="utf-8"))["benchmark_mode"],
            json.loads(path.read_text(encoding="utf-8"))["reasoning_trigger"],
        )
        for path in generated
    } == {
        ("dual", "terminal_once"),
        ("dual", "per_turn"),
    }


def test_mobilegym_outcome_evidence_uses_observable_judge_issues_only() -> None:
    module_path = Path("examples/mobilegym_agentic/app/agent.py")
    spec = importlib.util.spec_from_file_location("mobilegym_agent_for_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    evidence = module._mobilegym_outcome_evidence(
        {
            "id": "alipay.CheckBalance",
            "judge": {
                "success": False,
                "clean": True,
                "progress": 0.5,
                "issues": [{"field": "balance", "expected": 10, "actual": 9, "passed": False}],
            },
            "execution": {"stop_reason": "COMPLETE", "agent_message": "done", "steps": 1},
        },
        task_id="alipay.CheckBalance",
    )

    assert evidence["goal_checks"] == [{"field": "balance", "expected": 10, "actual": 9}]
    assert "success" not in json.dumps(evidence)
    assert "clean" not in json.dumps(evidence)
    assert "progress" not in json.dumps(evidence)


def test_mobilegym_wrapper_passes_sample_seed_as_a_subprocess_argument(tmp_path: Path, monkeypatch) -> None:
    module_path = Path("examples/mobilegym_agentic/app/agent.py")
    spec = importlib.util.spec_from_file_location("mobilegym_agent_seed_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    runs_root = tmp_path / "runs"
    mobilegym_repo = tmp_path / "mobilegym"
    mobilegym_repo.mkdir()
    monkeypatch.setenv("MOBILEGYM_PYTHON", "/venv/bin/python")
    monkeypatch.setenv("MOBILEGYM_REPO_DIR", str(mobilegym_repo))
    monkeypatch.setenv("MOBILEGYM_ENV_URL", "https://mobilegym.example")
    monkeypatch.setenv("MOBILEGYM_RUNS_ROOT", str(runs_root))
    monkeypatch.setenv("OPENAI_BASE_URL", "http://relax-session.example/v1")

    captured: dict = {}

    def _fake_run(cmd, **_kwargs):
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", _fake_run)
    module.run_mobilegym_episode("task.alpha", session_id="session-1", sample_seed=42)

    seed_index = captured["cmd"].index("--sample-seed") + 1
    assert captured["cmd"][seed_index] == "42"
    assert isinstance(captured["cmd"][seed_index], str)


def test_g1_result_rows_excludes_mobilegym_multiprocess_mirror(tmp_path: Path) -> None:
    module_path = Path("examples/mobilegym_agentic/g1_cpu_sweep.py")
    spec = importlib.util.spec_from_file_location("mobilegym_g1_for_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    row = {"id": "alipay.CheckBalance", "execution": {"stop_reason": "COMPLETE"}}
    top_level = tmp_path / "run" / "results.jsonl"
    mirrored = tmp_path / "run" / "shards" / "p00" / "results.jsonl"
    owned_shard = tmp_path / "shard-00" / "run" / "results.jsonl"
    for path in (top_level, mirrored, owned_shard):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    assert module._result_rows(tmp_path) == [row, row]


def test_g2_launchers_preserve_multimodal_training_inputs_and_debug_replay() -> None:
    rollout_script = Path("examples/mobilegym_agentic/run_mobilegym_e2e.sh").read_text(encoding="utf-8")
    actor_script = Path("examples/mobilegym_agentic/run_g2_actor_tp4_smoke.sh").read_text(encoding="utf-8")

    assert '--multimodal-keys \'{"image":"images"}\'' in rollout_script
    assert '--save-debug-rollout-data "${EXP_DIR}/debug_rollout/{rollout_id}.pt"' in rollout_script
    assert '--load-debug-rollout-data "${CAPTURE_PT}"' in actor_script
    assert '--multimodal-keys \'{"image":"images"}\'' in actor_script
    assert "--fully-async" not in actor_script


def test_g3_rollout8_mode_is_cross_node_inference_only_and_fail_closed() -> None:
    rollout_script = Path("examples/mobilegym_agentic/run_g3_rollout8_smoke.sh").read_text(encoding="utf-8")
    submit_script = Path("examples/mobilegym_agentic/submit_mobilegym_e2e.sh").read_text(encoding="utf-8")
    spmd_script = Path("scripts/entrypoint/spmd-multinode.sh").read_text(encoding="utf-8")

    assert '--resource "{\\"rollout\\":[1,${ROLLOUT_ENGINE_COUNT}]}"' in rollout_script
    assert 'ROLLOUT_ENGINE_COUNT="${ROLLOUT_ENGINE_COUNT:-8}"' in rollout_script
    assert "--num-gpus-per-node 4" in rollout_script
    assert "--rm-type dummy" in rollout_script
    assert "--use-health-check" not in rollout_script
    assert "RAY_CLUSTER_READY_TIMEOUT_S" in submit_script
    assert "FLASHINFER_WORKSPACE_BASE" in rollout_script
    assert "No locks available" in Path("examples/mobilegym_agentic/scripts/check_g3_rollout8.py").read_text(
        encoding="utf-8"
    )
    assert "cluster_ready_deadline" in spmd_script
    assert "RELAX_SPMD_COMPLETION_FILE" in submit_script
    assert "publish_spmd_completion" in spmd_script


def test_g4_full12_mode_is_two_publication_rounds_with_direct_gates() -> None:
    run_script = Path("examples/mobilegym_agentic/run_mobilegym_e2e.sh").read_text(encoding="utf-8")
    submit_script = Path("examples/mobilegym_agentic/submit_mobilegym_e2e.sh").read_text(encoding="utf-8")

    assert 'G4_FULL12_ONLY="${G4_FULL12_ONLY:-0}"' in submit_script
    assert 'export NUM_ROLLOUT="${NUM_ROLLOUT:-2}"' in submit_script
    assert "G4_FULL12_ONLY=1 requires exactly two publication rounds" in submit_script
    assert "--save-interval 1" in run_script
    assert "--distributed-timeout-minutes 5" in run_script
    assert "RELAX_REQUIRE_WEIGHT_PUBLICATION=1" in run_script
    assert "RELAX_DUAL_JUDGE_MARKER_DIR" in run_script
    assert "--allow-synchronized-multi-host-clock" in run_script
    assert "--multi-host-clock-max-offset-ms" in run_script
    assert "--ready-markers" in run_script
    assert "check_g4_full12.py" in run_script
    assert 'HOST_PYTHON="${HOST_PYTHON:-python3.11}"' in submit_script


def test_g5_full24_mode_matches_production_topology_and_requires_post_warmup_consumption() -> None:
    run_script = Path("examples/mobilegym_agentic/run_mobilegym_e2e.sh").read_text(encoding="utf-8")
    submit_script = Path("examples/mobilegym_agentic/submit_mobilegym_e2e.sh").read_text(encoding="utf-8")
    checker = Path("examples/mobilegym_agentic/scripts/check_g5_full24.py").read_text(encoding="utf-8")

    assert 'G5_FULL24_ONLY="${G5_FULL24_ONLY:-0}"' in submit_script
    assert "G5_FULL24_ONLY=1 requires exactly six nodes with four GPUs each" in submit_script
    assert "G5_FULL24_ONLY=1 requires at least three rounds (one warmup plus at least two measured)" in submit_script
    assert (
        "RESOURCE_JSON='{"
        + '"actor":[1,4],"rollout":[1,12],"advantages":[1,0],'
        + '"judge_accuracy":[1,4],"judge_multiturn_vlm":[1,4]}'
        in run_script
    )
    assert "--actor-num-nodes 1" in run_script
    assert "--actor-num-gpus-per-node 4" in run_script
    assert "--num-data-storage-units 8" in run_script
    assert "--per-rank-fetch" not in run_script
    assert "--expected-groups-per-round 8" in run_script
    assert "--expected-samples-per-group 8" in run_script
    assert '--measure-updates "$((NUM_ROLLOUT - 1))"' in run_script
    assert "capture_gpu_inventory.py" in submit_script
    assert (
        'FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/tmp/relax-flashinfer/${SLURM_JOB_ID}}"'
        in submit_script
    )
    assert "check_flashinfer_workspace.py" in submit_script
    assert '"FLASHINFER_WORKSPACE_BASE",' in run_script
    assert "FLASHINFER_WORKSPACE_BASE,RELAX_DUAL_JUDGE_MARKER_DIR" in run_script
    assert "RELAX_PLACEMENT_MANIFEST_DIR" in run_script
    assert "check_g5_full24.py" in run_script
    assert "no post-warmup policy version reached committed rollout" in checker
    assert "training pixel_values were not compacted to bfloat16" in checker
    assert "put_get_socket._put_to_single_storage_unit] attempt 1/2 failed" in checker


def test_flashinfer_workspace_probe_requires_local_locking_storage(tmp_path: Path) -> None:
    module_path = Path("examples/mobilegym_agentic/scripts/check_flashinfer_workspace.py")
    spec = importlib.util.spec_from_file_location("flashinfer_workspace_probe", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    invalid_workspace = tmp_path / "shared-cache"
    try:
        module.check_workspace(str(invalid_workspace))
    except RuntimeError as exc:
        assert "must be under /tmp/relax-flashinfer" in str(exc)
    else:
        raise AssertionError("non-local FlashInfer workspace unexpectedly accepted")

    module._REQUIRED_ROOT = tmp_path
    result = module.check_workspace(str(tmp_path / "job-42"))
    assert result["workspace"] == str(tmp_path / "job-42")
    assert result["free_bytes"] >= module._MIN_FREE_BYTES
    assert (tmp_path / "job-42" / ".flock_probe").is_file()


def test_g3_actor8_mode_is_cross_node_tp4_dp2_and_sync_debug_replay() -> None:
    actor_script = Path("examples/mobilegym_agentic/run_g3_actor_tp4_dp2_smoke.sh").read_text(encoding="utf-8")
    submit_script = Path("examples/mobilegym_agentic/submit_mobilegym_e2e.sh").read_text(encoding="utf-8")

    assert "--resource '{\"actor\":[1,8]}'" in actor_script
    assert "--actor-num-nodes 2" in actor_script
    assert "--actor-num-gpus-per-node 4" in actor_script
    assert "--tensor-model-parallel-size 4" in actor_script
    assert '--load-debug-rollout-data "${CAPTURE_PT}"' in actor_script
    assert "--load-debug-rollout-data-subsample" not in actor_script
    assert "--fully-async" not in actor_script
    assert "--use-health-check" not in actor_script
    assert "G3_ACTOR8_ONLY" in submit_script


def test_g3_dual8_mode_uses_production_tp2_judges_and_fail_closed_execution() -> None:
    run_script = Path("examples/mobilegym_agentic/run_mobilegym_e2e.sh").read_text(encoding="utf-8")
    submit_script = Path("examples/mobilegym_agentic/submit_mobilegym_e2e.sh").read_text(encoding="utf-8")

    assert "G3_DUAL8_ONLY" in run_script
    assert "G3_DUAL8_ONLY" in submit_script
    assert "RESOURCE_JSON='{" + '"rollout":[1,4],"judge_accuracy":[1,2],"judge_multiturn_vlm":[1,2]}' in run_script
    assert "check_g3_dual8.py" in run_script
    assert "--direct" in run_script


def test_g2_bootstrap_pins_cuda13_cudnn_and_runs_real_fused_attention_gate() -> None:
    setup_script = Path("examples/mobilegym_agentic/scripts/setup_relax_env.sh").read_text(encoding="utf-8")
    submit_script = Path("examples/mobilegym_agentic/submit_mobilegym_e2e.sh").read_text(encoding="utf-8")

    assert '"nvidia-cudnn-cu13==${CUDNN_VERSION}"' in setup_script
    assert 'CUBLAS_VERSION="${CUBLAS_VERSION:-13.0.0.19}"' in setup_script
    assert 'CUDA_NVRTC_VERSION="${CUDA_NVRTC_VERSION:-13.0.48}"' in setup_script
    assert '--no-deps "nvidia-cudnn-cu13==${CUDNN_VERSION}"' in setup_script
    assert '--no-deps --force-reinstall "nvidia-cudnn-cu13==${CUDNN_VERSION}"' in setup_script
    assert "actual_header_version == expected_header_version" in setup_script
    assert 'LD_PRELOAD="${cudnn_preload}${LD_PRELOAD:+:${LD_PRELOAD}}"' in setup_script
    assert "torch.backends.cudnn.version() == expected_cudnn_runtime" in setup_script
    assert setup_script.index('--force-reinstall "nvidia-cudnn-cu13==${CUDNN_VERSION}"') > setup_script.index(
        'install -q -e "${TRANSFER_QUEUE_DIR}"'
    )
    assert "nvidia-cudnn-cu12==" not in setup_script
    assert "NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=1 NVTE_UNFUSED_ATTN=0" in setup_script
    assert "loss.backward()" in setup_script
    assert "NVTE_F16_arbitrary_seqlen" in setup_script
    assert 'Path("/proc/self/maps")' in setup_script
    assert "g2_relax_env_te214_sm90_cuda13_v2" in submit_script
