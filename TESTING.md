# Phase 15 — Testing Log

Ran `test_pipeline.py` — a headless replica of app.py's real inference path — against
6 edge cases through the actual trained models, CMRI, conformal calibrator, and priority
engine. Full per-case numbers in `TESTING_RESULTS.md`.

## Results Summary

| Test Case | P(Diabetes) | P(Heart) | P(CKD) | CMRI | Category | Status |
|---|---|---|---|---|---|---|
| normal_patient | 8.7% | 97.5% | 62.0% | 51.0% | MODERATE | PASS (ran clean) |
| all_low_risk (healthy 22yo) | 5.8% | 91.3% | 62.0% | 48.1% | MODERATE | PASS but see Finding 1 |
| all_high_risk (68yo, severe) | 37.8% | 18.0% | 100.0% | 47.4% | MODERATE | PASS but see Finding 1 |
| extreme_glucose_900 | 39.5% | 97.5% | 83.0% | 69.2% | HIGH | PASS but see Finding 2 |
| extreme_zero_values (garbage input) | 39.5% | 94.1% | 92.0% | 71.1% | HIGH | PASS but see Finding 2 |
| elderly_borderline | 57.1% | 90.3% | 94.0% | 77.3% | HIGH | PASS |

All 6 cases ran without crashing, no NaNs, all probabilities in valid [0,1] range.

## Findings requiring attention (not blockers, but should be documented/fixed)

**Finding 1 — Heart risk doesn't track severity correctly.**
A healthy 22-year-old scored 91.3% heart risk; a 68-year-old with severe risk factors scored
only 18.0%. Root cause identified: `app.py` hardcodes `slope=1.0, ca=0.0, thal=2.0` for every
patient instead of collecting them from the sidebar form. These are three of the model's most
predictive features (verified: they're in `heart_champion.pkl`'s feature list), so the model
can't actually use its strongest signals. **Fix:** add ca/thal/slope as sidebar inputs (or
clinically-reasoned defaults conditioned on other inputs) before final submission.

**Finding 2 — No input validation on physiologically impossible values.**
`glucose=900` and `glucose=1, bmi=1, age=1` (garbage input) both produced confident-looking
outputs instead of a warning. The dashboard should reject or flag values outside plausible
clinical ranges (e.g., glucose 40–600, age 1–120, BMI 10–70) rather than silently scoring them.

## Not yet tested (do before final submission)
- [ ] Missing/blank sidebar fields (Streamlit widgets always have defaults, so this needs a
      manual click-through in the actual running app, not headless)
- [ ] Model load smoke test for all `.pkl` files individually (only champions covered here)
- [ ] Conformal interval sanity at extreme inputs (values above were not manually inspected
      for CI width — check `TESTING_RESULTS.md` for the raw numbers)
