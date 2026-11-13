"""Rule-based go / no-go memo.

The decision follows fixed thresholds, and every number in the text is
formatted straight from the engine output. No language model is involved.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Thresholds:
    min_ltv_to_cac: float = 3.0
    max_payback_months: int = 12
    max_cannibalisation_share: float = 0.50  # cannibalisation as a share of interchange


def decide(pnl: dict, required_sample: int, t: Thresholds | None = None) -> dict:
    """Apply the thresholds and return the decision with the checks behind it."""
    t = t or Thresholds()
    c = pnl["components"]
    cannibal_share = abs(c["cannibalisation"]) / c["interchange"] if c["interchange"] else float("inf")
    checks = {
        "Contribution is positive over the horizon": pnl["contribution"] > 0,
        f"Payback within {t.max_payback_months} months":
            pnl["payback_month"] is not None and pnl["payback_month"] <= t.max_payback_months,
        f"LTV to CAC at least {t.min_ltv_to_cac:.1f}": pnl["ltv_to_cac"] >= t.min_ltv_to_cac,
        f"Cannibalisation below {t.max_cannibalisation_share:.0%} of interchange":
            cannibal_share <= t.max_cannibalisation_share,
        "Base is large enough for the launch test": 2 * required_sample <= pnl["eligible_users"],
    }
    passed = sum(checks.values())
    if passed == len(checks):
        decision = "GO"
    elif checks["Contribution is positive over the horizon"] and passed >= len(checks) - 2:
        decision = "CONDITIONAL GO"
    else:
        decision = "NO-GO"
    return {"decision": decision, "checks": checks, "cannibalisation_share": cannibal_share}


def write_memo(pnl: dict, required_sample: int, t: Thresholds | None = None) -> str:
    """One-page memo in Markdown."""
    t = t or Thresholds()
    d = decide(pnl, required_sample, t)
    c = pnl["components"]
    payback = f"month {pnl['payback_month']}" if pnl["payback_month"] else "not reached in the horizon"
    lines = [
        f"## Launch decision: {d['decision']}",
        "",
        f"**Proposal:** cashback debit card at {pnl['cashback_rate']:.2%} cashback, "
        f"offered to {pnl['eligible_users']:,} existing users.",
        "",
        "**Headline numbers (simulated)**",
        f"- Expected adopters: {pnl['adopters']:,.0f} ({pnl['adoption_rate']:.1%} of the base)",
        f"- Contribution over the horizon: {pnl['contribution']:,.0f}",
        f"- Payback: {payback}",
        f"- LTV to CAC: {pnl['ltv_to_cac']:.1f}",
        "",
        "**Where the money comes from and goes**",
        f"- Interchange revenue: {c['interchange']:,.0f}",
        f"- Deposit halo: {c['deposit_halo']:,.0f}",
        f"- Cashback cost: {c['cashback']:,.0f}",
        f"- Cannibalised fee revenue: {c['cannibalisation']:,.0f} "
        f"({d['cannibalisation_share']:.0%} of interchange)",
        f"- Servicing, issuance and acquisition: {c['servicing'] + c['issuance'] + c['acquisition']:,.0f}",
        "",
        "**Checks**",
    ]
    lines += [f"- {'PASS' if ok else 'FAIL'}: {name}" for name, ok in d["checks"].items()]
    lines += [
        "",
        "**Launch test**",
        f"- Run target vs control with at least {required_sample:,} users per group before full rollout.",
        "",
        "**Kill criteria**",
        "- Stop if the test shows no significant lift in adoption over control.",
        "- Stop if the split between groups fails the sample ratio check.",
        f"- Stop if measured cannibalisation exceeds {t.max_cannibalisation_share:.0%} of interchange.",
        "- Stop if cardholder churn in the first three months is above the assumed rate.",
        "",
        "_All figures come from a simulation on synthetic data with illustrative assumptions._",
    ]
    return "\n".join(lines)
