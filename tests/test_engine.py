"""Tests for the deterministic engine. Run with: pytest"""
import re

import numpy as np
import pytest

from data.generate_ecosystem import generate_users
from engine import experiment as ex
from engine import unit_economics as ue
from engine.memo import decide, write_memo


@pytest.fixture(scope="module")
def users():
    return generate_users(20_000, seed=1)


def test_generator_is_reproducible():
    a, b = generate_users(1_000, seed=5), generate_users(1_000, seed=5)
    assert a.equals(b)
    assert len(a) == 1_000 and a["user_id"].is_unique


def test_adoption_rises_with_cashback(users):
    low = ue.adoption_probability(users, 0.002).mean()
    high = ue.adoption_probability(users, 0.015).mean()
    assert 0 <= low < high <= 0.95


def test_negative_cashback_rejected(users):
    with pytest.raises(ValueError):
        ue.adoption_probability(users, -0.01)


def test_pnl_is_deterministic_and_components_sum(users):
    first, second = ue.launch_pnl(users, 0.01), ue.launch_pnl(users, 0.01)
    assert first == second
    assert first["contribution"] == pytest.approx(sum(first["components"].values()))


def test_segments_sum_to_total(users):
    total = ue.launch_pnl(users, 0.01)["contribution"]
    by_segment = ue.segment_breakdown(users, 0.01)["contribution"].sum()
    assert by_segment == pytest.approx(total)


def test_cashback_cap_limits_cost(users):
    capped = ue.launch_pnl(users, 0.05, ue.Assumptions(cashback_cap=1.0))
    uncapped = ue.launch_pnl(users, 0.05, ue.Assumptions(cashback_cap=1e9))
    assert abs(capped["components"]["cashback"]) < abs(uncapped["components"]["cashback"])


def test_margin_dilution_is_caught(users):
    """Very high cashback with thin interchange must lose money and fail the memo."""
    thin = ue.Assumptions(interchange_rate=0.002, cashback_cap=1e9)
    pnl = ue.launch_pnl(users, 0.03, thin)
    assert pnl["contribution"] < 0
    assert pnl["payback_month"] is None
    assert decide(pnl, required_sample=1_000)["decision"] == "NO-GO"


def test_best_cashback_is_the_sweep_maximum(users):
    sweep = ue.sweep_cashback(users, np.linspace(0, 0.02, 21))
    best = ue.best_cashback(sweep)
    assert sweep["contribution"].max() == sweep.loc[sweep["cashback_rate"] == best, "contribution"].iloc[0]


def test_sample_size_matches_textbook_value():
    # 10% baseline, +2 points, 5% alpha, 80% power: 3,839 per group by the standard formula.
    assert ex.sample_size_per_group(0.10, 0.02) == 3839


def test_smaller_effect_needs_more_users():
    assert ex.sample_size_per_group(0.10, 0.01) > ex.sample_size_per_group(0.10, 0.02)


def test_mde_is_inverse_of_sample_size():
    n = ex.sample_size_per_group(0.10, 0.02)
    assert ex.minimum_detectable_effect(n, 0.10) == pytest.approx(0.02, abs=1e-4)


def test_srm_detection():
    assert ex.srm_check(5_000, 5_000)["mismatch"] is False
    assert ex.srm_check(5_000, 4_500)["mismatch"] is True


def test_two_proportion_test():
    clear = ex.two_proportion_test(600, 5_000, 500, 5_000)
    assert clear["significant"] and clear["ci_low"] > 0
    none = ex.two_proportion_test(500, 5_000, 500, 5_000)
    assert not none["significant"] and none["absolute_lift"] == 0


def test_simulated_test_detects_a_real_lift(users):
    result = ex.simulate_test(users, target_rate=0.012, control_rate=0.004, n_per_group=4_000)
    assert result["significant"] and result["absolute_lift"] > 0
    assert result["srm"]["mismatch"] is False


def test_memo_numbers_match_engine(users):
    """Every headline number in the memo must equal the engine output."""
    pnl = ue.launch_pnl(users, 0.01)
    memo = write_memo(pnl, required_sample=3_839)
    assert f"{pnl['contribution']:,.0f}" in memo
    assert f"{pnl['adopters']:,.0f}" in memo
    assert f"{pnl['ltv_to_cac']:.1f}" in memo
    assert "3,839" in memo
    assert re.search(r"Launch decision: (GO|CONDITIONAL GO|NO-GO)", memo)
