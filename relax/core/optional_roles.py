# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Optional role registration helpers for controller-side wiring."""

from argparse import Namespace


GENRM_ROLE = "genrm"
DEDICATED_JUDGE_ROLES = ("judge_accuracy", "judge_multiturn_vlm")


def register_genrm(config: Namespace, algo: dict) -> list[str]:
    """Conditionally register GenRM into the algo dict."""
    if getattr(config, "genrm_model_path", None) is None:
        return []

    from relax.components.genrm import GenRM

    algo[GENRM_ROLE] = GenRM
    return [GENRM_ROLE]


def register_sft_rollout(config: Namespace, algo: dict) -> list:
    """Conditionally register Rollout into the SFT algo dict.

    SFT mode is identified by ``loss_type == "sft"`` throughout the codebase.
    Do not switch this to ``advantage_estimator == "sft"``: Megatron's parser
    does not accept ``"sft"`` as an ``--advantage-estimator`` choice.

    Rollout is only spun up when ``--sft-predict-interval`` is set — that is
    the sole consumer of generative eval under SFT today.
    """
    if getattr(config, "loss_type", None) != "sft":
        return []
    if getattr(config, "sft_predict_interval", None) is None:
        return []

    from relax.components.rollout import Rollout
    from relax.core.registry import ROLES

    algo[ROLES.rollout] = Rollout
    return [ROLES.rollout]


def register_dual_judges(config: Namespace, algo: dict) -> list[str]:
    if getattr(config, "judge_services", None) is None:
        return []
    from relax.components.genrm import GenRM

    for role in DEDICATED_JUDGE_ROLES:
        algo[role] = GenRM
    return list(DEDICATED_JUDGE_ROLES)


def register_extra_roles(config: Namespace, algo: dict) -> list:
    """Register optional roles and return them in controller iteration
    order."""
    extras = []
    extras.extend(register_genrm(config, algo))
    extras.extend(register_dual_judges(config, algo))
    extras.extend(register_sft_rollout(config, algo))
    return extras
