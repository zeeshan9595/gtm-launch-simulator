"""Generate a synthetic multi-product fintech customer base.

Everything here is simulated. No real customer, company or market data is used.
The distributions are illustrative choices, documented in the README.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEGMENTS = ["mass", "engaged", "affluent"]
SEGMENT_SHARE = [0.60, 0.30, 0.10]
CHANNELS = ["organic", "paid", "cross_sell"]
CHANNEL_SHARE = [0.45, 0.30, 0.25]

# Per segment: probability of holding each product, typical balances and spend.
PROFILE = {
    "mass":     dict(deposit=0.35, bnpl=0.15, p2p=0.70, qr=0.30, balance=180.0,  fees=0.60, spend=120.0),
    "engaged":  dict(deposit=0.65, bnpl=0.35, p2p=0.90, qr=0.60, balance=650.0,  fees=1.80, spend=320.0),
    "affluent": dict(deposit=0.90, bnpl=0.25, p2p=0.85, qr=0.70, balance=4200.0, fees=3.50, spend=780.0),
}


def generate_users(n_users: int = 100_000, seed: int = 42) -> pd.DataFrame:
    """Return one row per simulated user. Same seed gives the same base."""
    if n_users <= 0:
        raise ValueError("n_users must be positive")
    rng = np.random.default_rng(seed)

    segment = rng.choice(SEGMENTS, size=n_users, p=SEGMENT_SHARE)
    channel = rng.choice(CHANNELS, size=n_users, p=CHANNEL_SHARE)
    tenure = rng.integers(1, 61, size=n_users)

    def per_segment(key: str) -> np.ndarray:
        """Look up one PROFILE value for every user's segment."""
        values = np.array([PROFILE[s][key] for s in SEGMENTS])
        return values[pd.Categorical(segment, categories=SEGMENTS).codes]

    holds_deposit = rng.random(n_users) < per_segment("deposit")
    holds_bnpl = rng.random(n_users) < per_segment("bnpl")
    uses_p2p = rng.random(n_users) < per_segment("p2p")
    uses_qr = rng.random(n_users) < per_segment("qr")

    # Lognormal noise keeps values positive and right-skewed, like real balances.
    deposit_balance = np.where(
        holds_deposit, per_segment("balance") * rng.lognormal(0.0, 0.6, n_users), 0.0
    )
    activity = uses_p2p.astype(float) + uses_qr.astype(float)
    existing_fee_revenue = per_segment("fees") * activity / 2 * rng.lognormal(0.0, 0.4, n_users)
    card_spend_potential = per_segment("spend") * rng.lognormal(0.0, 0.5, n_users)

    n_products = (
        holds_deposit.astype(int) + holds_bnpl.astype(int)
        + uses_p2p.astype(int) + uses_qr.astype(int)
    )
    # Fewer products and shorter tenure mean higher churn risk.
    churn_risk = np.clip(0.35 - 0.05 * n_products - 0.002 * tenure + rng.normal(0, 0.04, n_users), 0.01, 0.60)

    return pd.DataFrame({
        "user_id": np.arange(1, n_users + 1),
        "segment": segment,
        "channel": channel,
        "tenure_months": tenure,
        "holds_deposit": holds_deposit,
        "holds_bnpl": holds_bnpl,
        "uses_p2p": uses_p2p,
        "uses_qr": uses_qr,
        "n_products": n_products,
        "deposit_balance": deposit_balance.round(2),
        "existing_fee_revenue": existing_fee_revenue.round(4),
        "card_spend_potential": card_spend_potential.round(2),
        "churn_risk": churn_risk.round(4),
    })


if __name__ == "__main__":
    users = generate_users()
    users.to_csv("data/ecosystem.csv", index=False)
    print(f"Wrote {len(users):,} simulated users to data/ecosystem.csv")
    print(users.groupby("segment")[["deposit_balance", "existing_fee_revenue", "card_spend_potential"]].mean().round(2))
