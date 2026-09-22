from src.metrics import (
    risk_propagation_rate_from_all_agents,
    consensus_robustness,
    unsafe_consensus_rate,
    recovery,
    fsu,
)


def test_rpr_no_propagation():
    # a1 exposed; a2/a3 initially safe; only a1 unsafe
    r = risk_propagation_rate_from_all_agents(
        ["a1", "a2", "a3"],
        ["a1"],
        ["a1"],
    )
    assert r == 0.0


def test_rpr_half_propagation():
    # a1 exposed; a2/a3 initially safe; a2 becomes unsafe
    r = risk_propagation_rate_from_all_agents(
        ["a1", "a2", "a3"],
        ["a1"],
        ["a1", "a2"],
    )
    assert r == 0.5


def test_rpr_full_propagation():
    # a1 exposed; both previously safe agents become unsafe
    r = risk_propagation_rate_from_all_agents(
        ["a1", "a2", "a3"],
        ["a1"],
        ["a1", "a2", "a3"],
    )
    assert r == 1.0


def test_consensus_metrics_safe():
    assert consensus_robustness(False) == 1.0
    assert unsafe_consensus_rate(False) == 0.0


def test_consensus_metrics_unsafe():
    assert consensus_robustness(True) == 0.0
    assert unsafe_consensus_rate(True) == 1.0


def test_recovery():
    assert recovery(True, False) == 1.0
    assert recovery(True, True) == 0.0
    assert recovery(False, False) is None


def test_fsu():
    assert abs(fsu(1.0, 1.0) - 1.0) < 1e-9
    assert fsu(0.0, 1.0) == 0.0
    assert fsu(1.0, 0.0) == 0.0
