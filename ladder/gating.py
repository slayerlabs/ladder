"""User-specified reporting policy, not an empirically measured capability map."""
RUNGS = (8, 25, 50, 120)
POLICY = {
    "corpus_bpb_all_slices": ("primary", "primary", "primary", "primary"),
    "agreement_pairs": ("active", "active", "active", "active"),
    "case_negation_gender_pairs": ("floor", "active", "active", "active"),
    "long_distance_islands": ("floor", "floor", "weak", "active"),
    "mc_correct_choice_bpb": ("floor", "weak", "active", "active"),
    "mc_accuracy": ("off", "off", "off", "reportable"),
    "ewok_knowledge": ("off", "off", "off", "canary"),
}
SUPPORT = {
    "corpus_bpb_all_slices": "standalone frozen-corpus scorer; real corpus awaiting inputs",
    "agreement_pairs": "sentence and critical-region scorers; MultiBLiMP-PL pool awaiting native review",
    "case_negation_gender_pairs": "not implemented",
    "long_distance_islands": "not implemented",
    "mc_correct_choice_bpb": "standalone correct-continuation scorer; MC data awaiting inputs",
    "mc_accuracy": "legacy synthetic diagnostic only; natural MC accuracy adapter pending",
    "ewok_knowledge": "not implemented",
}


def scale_policy(rung):
    if rung not in RUNGS:
        raise ValueError(f"Rung must be one of {RUNGS} million parameters; no interpolation")
    column = RUNGS.index(rung)
    return {"version": 1, "rung_millions": rung, "source": "user-specified policy; uncalibrated",
            "metrics": {key: {"status": values[column], "implementation": SUPPORT[key]}
                        for key, values in POLICY.items()}}


def resolve_rung(parameters, rung=None):
    if rung is not None:
        scale_policy(rung)
        return rung
    if parameters in tuple(n * 1_000_000 for n in RUNGS):
        return parameters // 1_000_000
    return None
