import numpy as np
import pytest
from pydantic import ValidationError

from invariant.benchmark.config import SyntheticMarketConfig, load_benchmark_config
from invariant.benchmark.ground_truth import (
    CausalEdge,
    GroundTruth,
    Mechanism,
    NodeRole,
    SignalTruth,
)
from invariant.benchmark.synthetic import generate_market
from invariant.core.types import StageName, Verdict

MEAN = frozenset({Mechanism.LINEAR, Mechanism.LINEAR_TIME_VARYING})
VAR = Mechanism.VARIANCE


def _cfg(preset: str | None = None, **updates: object) -> SyntheticMarketConfig:
    return load_benchmark_config(preset).model_copy(update={"n_obs": 300, "burn_in": 10, **updates})


def _gt(preset: str | None = None, **updates: object) -> GroundTruth:
    return generate_market(_cfg(preset, **updates), 11).ground_truth


def test_json_round_trip_is_exact() -> None:
    gt = _gt("multi_asset")
    assert GroundTruth.model_validate_json(gt.model_dump_json()) == gt


def test_ground_truth_is_frozen() -> None:
    gt = _gt()
    with pytest.raises(ValidationError):
        gt.seed = 3  # type: ignore[misc]


def test_records_seed_config_and_timing() -> None:
    cfg = _cfg()
    gt = generate_market(cfg, 11).ground_truth
    assert gt.seed == 11
    assert gt.config == cfg
    assert "returns[t+1]" in gt.timing_convention


def test_expected_verdicts_for_default_config() -> None:
    gt = _gt()
    table = {
        s.signal_id: (s.has_causal_edge, s.expected_verdict, s.expected_kill_stage, s.identifiable)
        for s in gt.signals
    }
    assert table == {
        "s_real": (True, Verdict.PASS, None, True),
        "s_conf": (False, Verdict.KILL, StageName.CAUSAL, True),
        "s_decay": (True, Verdict.KILL, StageName.DECAY, True),
        "s_noise": (False, Verdict.KILL, StageName.CAUSAL, True),
    }


def test_hidden_confounder_marks_s_conf_unidentifiable_with_nonzero_bias() -> None:
    gt = _gt("hidden_confounder")
    conf = gt.signal("s_conf")
    assert not conf.identifiable
    assert conf.expected_verdict is Verdict.KILL
    assert conf.partial_ic[0] > 0.03
    assert "causal sufficiency" in conf.note
    assert not gt.node("z").observed


def test_partial_ic_is_exactly_zero_for_non_causes_when_z_observed() -> None:
    gt = _gt("multi_asset")
    for sid in ("s_conf", "s_noise"):
        s = gt.signal(sid)
        np.testing.assert_allclose(s.partial_ic, 0.0, atol=1e-12)
    # ...but S_conf is a real trap: marginally correlated with forward returns.
    assert min(gt.signal("s_conf").marginal_ic) > 0.02
    np.testing.assert_allclose(gt.signal("s_noise").marginal_ic, 0.0, atol=1e-12)


def test_zero_ic_turns_a_signal_into_a_null() -> None:
    cfg = _cfg()
    gt = _gt(real=cfg.real.model_copy(update={"ic": 0.0}))
    s = gt.signal("s_real")
    assert (s.has_causal_edge, s.expected_verdict, s.expected_kill_stage) == (
        False,
        Verdict.KILL,
        StageName.CAUSAL,
    )
    assert ("s_real", 1) not in gt.parents("asset_00")


def test_decay_fields() -> None:
    cfg = load_benchmark_config()
    gt = generate_market(cfg, 1).ground_truth
    s = gt.signal("s_decay")
    assert s.coef_half_life == 250
    assert s.te_half_life_approx == 125
    assert s.final_coef_fraction == pytest.approx(2 ** (-(cfg.n_obs - 1 - 3600) / 250))
    assert gt.signal("s_real").coef_half_life is None
    # Effective IC reflects the average coefficient, so it is smaller than the initial one.
    assert s.effective_partial_ic[0] < s.partial_ic[0]
    assert s.expected_t[0] == pytest.approx(s.effective_partial_ic[0] * np.sqrt(cfg.n_obs - 1))


def test_parents_and_adjacency_single_asset() -> None:
    gt = _gt()
    assert gt.parents("asset_00") == {
        ("z", 1),
        ("s_real", 1),
        ("s_decay", 1),
        ("asset_00", 1),
    }
    assert gt.parents("asset_00", mechanisms=MEAN) == {("z", 1), ("s_real", 1), ("s_decay", 1)}
    assert gt.parents("asset_00", mechanisms=frozenset({VAR})) == {("asset_00", 1)}
    assert gt.parents("s_conf") == {("z", 1), ("s_conf", 1)}
    assert gt.parents("s_noise") == {("s_noise", 1)}
    names = gt.node_names
    adj = gt.adjacency(1)
    assert adj.shape == (len(names), len(names))
    col = names.index("asset_00")
    assert {names[i] for i in np.flatnonzero(adj[:, col])} == {"z", "s_real", "s_decay", "asset_00"}
    # Neither non-cause has any outgoing edge except its own persistence.
    for sid in ("s_conf", "s_noise"):
        row = names.index(sid)
        assert {names[j] for j in np.flatnonzero(adj[row])} == {sid}
    assert not gt.adjacency(0).any()
    assert not gt.adjacency(2).any()
    assert gt.max_lag == 1


def test_multi_asset_graph_has_latent_market_at_lag_zero() -> None:
    gt = _gt("multi_asset")
    assert gt.node("market").role is NodeRole.LATENT
    assert not gt.node("market").observed
    for asset in gt.asset_names:
        assert ("market", 0) in gt.parents(asset)
    adj0 = gt.adjacency(0)
    assert adj0.sum() == len(gt.asset_names)
    assert ("market", 1) in gt.parents("market", mechanisms=frozenset({VAR}))


def test_edge_coefficients_match_loadings() -> None:
    gt = _gt("multi_asset")
    for i, asset in enumerate(gt.asset_names):
        coefs = {e.source: e.coefficient for e in gt.edges if e.target == asset and e.lag == 1}
        assert coefs["z"] == gt.loadings.beta_z[i]
        assert coefs["s_real"] == gt.loadings.beta_real[i]
        assert coefs["s_decay"] == gt.loadings.beta_decay_initial[i]
        assert coefs[asset] == gt.config.garch.alpha
    decay_edges = [e for e in gt.edges if e.source == "s_decay" and e.target != "s_decay"]
    assert all(e.mechanism is Mechanism.LINEAR_TIME_VARYING for e in decay_edges)


# --- validators -------------------------------------------------------------------------------


def test_rejects_edge_to_unknown_node() -> None:
    gt = _gt()
    bad = CausalEdge(
        source="ghost", target="asset_00", lag=1, mechanism=Mechanism.LINEAR, coefficient=1.0
    )
    with pytest.raises(ValidationError, match="unknown node"):
        GroundTruth.model_validate({**gt.model_dump(), "edges": [*gt.model_dump()["edges"], bad]})


def test_rejects_duplicate_edges() -> None:
    data = _gt().model_dump()
    data["edges"] = [*data["edges"], data["edges"][0]]
    with pytest.raises(ValidationError, match="duplicate edges"):
        GroundTruth.model_validate(data)


def test_rejects_missing_signal() -> None:
    data = _gt().model_dump()
    data["signals"] = data["signals"][:-1]
    with pytest.raises(ValidationError, match="candidate"):
        GroundTruth.model_validate(data)


def test_signal_truth_kill_stage_consistency() -> None:
    base = _gt().signal("s_real").model_dump()
    with pytest.raises(ValidationError, match="kill_stage"):
        SignalTruth.model_validate({**base, "expected_kill_stage": StageName.CAUSAL})
    with pytest.raises(ValidationError, match="kill_stage"):
        SignalTruth.model_validate({**base, "expected_verdict": Verdict.KILL})


def test_signal_truth_decay_fields_all_or_nothing() -> None:
    base = _gt().signal("s_real").model_dump()
    with pytest.raises(ValidationError, match="decay fields"):
        SignalTruth.model_validate({**base, "coef_half_life": 10.0})


def test_signal_truth_rejects_bad_id() -> None:
    base = _gt().signal("s_real").model_dump()
    with pytest.raises(ValidationError, match="signal id"):
        SignalTruth.model_validate({**base, "signal_id": "Bad Id"})


def test_unknown_lookups_raise() -> None:
    gt = _gt()
    with pytest.raises(KeyError):
        gt.signal("s_nope")
    with pytest.raises(KeyError):
        gt.node("nope")
    with pytest.raises(KeyError):
        gt.parents("nope")
