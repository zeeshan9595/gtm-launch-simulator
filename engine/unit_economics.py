"""Deterministic unit economics for a cashback debit card launch.

Every figure the app shows comes from the functions in this file.
Nothing is random here: adoption is an expected value, so the same inputs
always give the same outputs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Assumptions:
    """Illustrative assumptions. Change them in the app sidebar."""

    interchange_rate: float = 0.012       # revenue earned on each unit of card spend
    cashback_cap: float = 15.0            # maximum cashback per user per month
    cannibalisation_rate: float = 0.15    # share of existing fee revenue lost per adopter
    deposit_halo: float = 0.08            # uplift in deposit balance for adopters
    net_interest_margin: float = 0.03     # annual margin earned on deposits
    servicing_cost: float = 0.40          # cost per active card per month
    issuance_cost: float = 3.00           # one-off cost per card issued
    monthly_card_churn: float = 0.03      # share of cardholders who stop each month
    spend_lift_per_pp: float = 0.10       # extra card spend per 1 percentage point of cashback
    horizon_months: int = 12

    def __post_init__(self) -> None:
        if not 0 <= self.monthly_card_churn < 1:
            raise ValueError("monthly_card_churn must be in [0, 1)")
        if self.horizon_months < 1:
            raise ValueError("horizon_months must be at least 1")


# Adoption curve per segment: (ceiling, midpoint cashback rate, steepness).
ADOPTION_CURVE = {
    "mass":     (0.30, 0.010, 350.0),
    "engaged":  (0.50, 0.008, 400.0),
    "affluent": (0.40, 0.006, 300.0),
}
# One-off cost to acquire one cardholder, by the channel the user came from.
ACQUISITION_COST = {"organic": 1.0, "cross_sell": 2.0, "paid": 6.0}


def adoption_probability(users: pd.DataFrame, cashback_rate: float) -> np.ndarray:
    """Expected probability that each user takes the card at this cashback rate.

    A logistic (S-shaped) curve per segment: adoption is low at low cashback,
    rises fastest around the midpoint, then flattens at the segment ceiling.
    Users holding more products are assumed more likely to adopt.
    """
    if cashback_rate < 0:
        raise ValueError("cashback_rate cannot be negative")
    ceiling = users["segment"].map(lambda s: ADOPTION_CURVE[s][0]).to_numpy(float)
    midpoint = users["segment"].map(lambda s: ADOPTION_CURVE[s][1]).to_numpy(float)
    steep = users["segment"].map(lambda s: ADOPTION_CURVE[s][2]).to_numpy(float)
    curve = ceiling / (1.0 + np.exp(-steep * (cashback_rate - midpoint)))
    engagement = 0.70 + 0.15 * users["n_products"].to_numpy(float)
    return np.clip(curve * engagement, 0.0, 0.95)


def _survival_months(a: Assumptions) -> np.ndarray:
    """Share of the launch cohort still active in each month of the horizon."""
    return (1.0 - a.monthly_card_churn) ** np.arange(a.horizon_months)


def _per_adopter_monthly(users: pd.DataFrame, cashback_rate: float, a: Assumptions) -> dict[str, np.ndarray]:
    """Monthly margin components for one active cardholder, per user row."""
    spend = users["card_spend_potential"].to_numpy(float) * (1.0 + a.spend_lift_per_pp * cashback_rate * 100)
    return {
        "interchange": spend * a.interchange_rate,
        "cashback": -np.minimum(spend * cashback_rate, a.cashback_cap),
        "cannibalisation": -users["existing_fee_revenue"].to_numpy(float) * a.cannibalisation_rate,
        "deposit_halo": users["deposit_balance"].to_numpy(float) * a.deposit_halo * a.net_interest_margin / 12,
        "servicing": -np.full(len(users), a.servicing_cost),
    }


def launch_pnl(users: pd.DataFrame, cashback_rate: float, a: Assumptions | None = None) -> dict:
    """Profit and loss of the launch cohort over the horizon."""
    a = a or Assumptions()
    p = adoption_probability(users, cashback_rate)
    monthly = _per_adopter_monthly(users, cashback_rate, a)
    survival = _survival_months(a)
    active_months = float(survival.sum())

    adopters = float(p.sum())
    components = {name: float((p * values).sum() * active_months) for name, values in monthly.items()}
    issuance = -adopters * a.issuance_cost
    acquisition = -float((p * users["channel"].map(ACQUISITION_COST).to_numpy(float)).sum())
    components["issuance"] = issuance
    components["acquisition"] = acquisition

    recurring_month_one = float(sum((p * v).sum() for v in monthly.values()))
    one_off = issuance + acquisition
    cumulative = one_off + recurring_month_one * np.cumsum(survival)
    positive = np.nonzero(cumulative >= 0)[0]
    payback_month = int(positive[0]) + 1 if len(positive) else None

    contribution = float(sum(components.values()))
    recurring_total = contribution - one_off
    ltv = recurring_total / adopters if adopters else 0.0
    cac = -one_off / adopters if adopters else 0.0
    return {
        "cashback_rate": cashback_rate,
        "eligible_users": int(len(users)),
        "adopters": adopters,
        "adoption_rate": adopters / len(users),
        "components": components,
        "contribution": contribution,
        "payback_month": payback_month,
        "ltv_per_adopter": ltv,
        "cac_per_adopter": cac,
        "ltv_to_cac": ltv / cac if cac else float("inf"),
        "cumulative_by_month": cumulative.tolist(),
    }


def sweep_cashback(users: pd.DataFrame, rates: np.ndarray, a: Assumptions | None = None) -> pd.DataFrame:
    """Adoption and contribution at each cashback rate (the elasticity curve)."""
    rows = []
    for rate in rates:
        r = launch_pnl(users, float(rate), a)
        rows.append({
            "cashback_rate": float(rate),
            "adoption_rate": r["adoption_rate"],
            "adopters": r["adopters"],
            "contribution": r["contribution"],
            "ltv_to_cac": r["ltv_to_cac"],
        })
    return pd.DataFrame(rows)


def best_cashback(sweep: pd.DataFrame) -> float:
    """Cashback rate with the highest contribution in the sweep."""
    return float(sweep.loc[sweep["contribution"].idxmax(), "cashback_rate"])


def segment_breakdown(users: pd.DataFrame, cashback_rate: float, a: Assumptions | None = None) -> pd.DataFrame:
    """Launch P&L per segment. Segment contributions sum to the total."""
    rows = []
    for segment, group in users.groupby("segment", sort=True):
        r = launch_pnl(group, cashback_rate, a)
        rows.append({
            "segment": segment,
            "users": len(group),
            "adoption_rate": r["adoption_rate"],
            "cannibalisation": r["components"]["cannibalisation"],
            "deposit_halo": r["components"]["deposit_halo"],
            "contribution": r["contribution"],
            "contribution_per_adopter": r["contribution"] / r["adopters"] if r["adopters"] else 0.0,
        })
    return pd.DataFrame(rows)
