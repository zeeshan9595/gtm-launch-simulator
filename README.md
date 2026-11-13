# GTM Launch Simulator

Should a fintech launch a cashback debit card to its existing customers, and at what cashback rate?

This project answers that on a simulated 100,000-user multi-product base. It models adoption, unit economics, cannibalisation of existing fee revenue, the deposit "halo", and the size of the target vs control test needed before rollout. A rule-based memo then returns Go, Conditional Go or No-Go.

> **Synthetic data.** Every user, balance and rate here is simulated, and every assumption is illustrative. No real customer, company or market data is used. The value is in the method, not the numbers.

## What the default run shows

| Cashback rate | Adoption | 12-month contribution | Payback | LTV to CAC | Decision |
| --- | --- | --- | --- | --- | --- |
| 0.5% (best in sweep) | 8.1% | 176,626 | Month 3 | 4.8 | Go |
| 1.0% | 22.9% | -19,255 | Not reached | 0.9 | No-Go |

Doubling the cashback nearly triples adoption and still destroys the business case: cashback cost grows faster than interchange. At 1.0% only the affluent segment is profitable, because its deposit halo outweighs the cashback paid.

## How it works

```
data/generate_ecosystem.py   Simulated users: segment, channel, products held, balances, spend
engine/unit_economics.py     Adoption curve, P&L components, payback, LTV to CAC, segment view
engine/experiment.py         Sample size, minimum detectable effect, sample ratio check, significance test
engine/memo.py               Thresholds and the Go / No-Go memo
app.py                       Streamlit workbench with five tabs
tests/test_engine.py         15 tests on the engine and memo
```

**Contribution per active cardholder per month**

```
  interchange       card spend x interchange rate
- cashback          min(card spend x cashback rate, monthly cap)
- cannibalisation   existing fee revenue x cannibalisation rate
+ deposit halo      deposit balance x uplift x net interest margin / 12
- servicing         fixed cost per active card
```

Issuance and acquisition are one-off costs. Cardholders churn at a fixed monthly rate, so the cohort shrinks over the 12-month horizon. Adoption follows an S-shaped (logistic) curve per segment.

**Design choices**

- The engine is deterministic: adoption is an expected value, so the same inputs always return the same outputs. The only random step is the simulated launch test, and it is seeded.
- The memo is rule-based. Each number in it is formatted straight from the engine output, and a test checks that they match.
- No language model is used anywhere.

## Run it

```bash
pip install -r requirements.txt
pytest                      # 15 tests
streamlit run app.py        # opens the workbench
```

## Limitations

- Adoption curves, costs and rates are assumptions, not estimates fitted to data.
- Cannibalisation and halo are fixed rates per adopter; in practice they would be measured with a holdout.
- Churn is constant and does not depend on the cashback rate.
- One launch cohort only: no later cohorts, no competitor response, no funding or regulatory cost.
