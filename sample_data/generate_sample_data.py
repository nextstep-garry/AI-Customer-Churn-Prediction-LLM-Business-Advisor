"""Generate SYNTHETIC demo datasets in the raw churn schema. Not real customers.

    python sample_data/generate_sample_data.py

Writes (deterministic, seed 42):
    churn_sample_dirty.csv       ~2,000 labelled rows with the same kinds of dirt the real raw file
                                 had: duplicate rows, typo'd countries, sentinel/invalid values
    churn_sample_unlabeled.csv   500 clean rows without the Exited column (scoring-only upload)

The label is drawn from a plausible churn-style rule (age, inactivity, Germany, product count),
so dashboards look realistic -- but it is invented, so do not read business conclusions from it.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent


def make_clean(n: int, rng: np.random.Generator, labelled: bool) -> pd.DataFrame:
    age = np.clip(18 + rng.gamma(4.0, 5.3, n), 18, 92).round().astype(int)
    geo = rng.choice(["France", "Germany", "Spain"], n, p=[0.50, 0.25, 0.25])
    products = rng.choice([1, 2, 3, 4], n, p=[0.52, 0.45, 0.025, 0.005])
    active = rng.binomial(1, 0.51, n)
    balance = np.where(rng.random(n) < 0.36, 0.0, np.clip(rng.normal(120000, 30000, n), 1, 250000)).round(2)
    df = pd.DataFrame({
        "RowNumber": np.arange(1, n + 1),
        "CustomerId": rng.choice(np.arange(15_500_000, 15_900_000), n, replace=False),
        "Surname": rng.choice(["Hargrave", "Hill", "Onio", "Boni", "Mitchell", "Chu", "Bartlett", "Obinna",
                               "He", "Bearce", "Andrews", "Kay", "Yin", "Kodilinyechukwu", "Maclean"], n),
        "CreditScore": np.clip(rng.normal(650, 96, n), 350, 850).round().astype(int),
        "Geography": geo,
        "Gender": rng.choice(["Female", "Male"], n, p=[0.45, 0.55]),
        "Age": age,
        "Tenure": rng.integers(0, 11, n),
        "Balance": balance,
        "NumOfProducts": products,
        "HasCrCard": rng.binomial(1, 0.70, n),
        "IsActiveMember": active,
        "EstimatedSalary": rng.uniform(11.58, 199992, n).round(2),
    })
    if labelled:
        logit = (-3.05 + 0.075 * (age - 18) + 1.1 * (geo == "Germany") - 1.0 * active
                 + 2.4 * (products >= 3) - 0.9 * (products == 2) + 0.25 * (df["Gender"] == "Female"))
        df["Exited"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


def make_dirty(rng: np.random.Generator) -> pd.DataFrame:
    df = make_clean(2000, rng, labelled=True).astype({"Balance": float, "EstimatedSalary": float,
                                                     "NumOfProducts": float, "Age": float, "CreditScore": float,
                                                     "HasCrCard": float})
    pick = lambda k: rng.choice(len(df), k, replace=False)  # noqa: E731
    for i, bad in zip(pick(12), ["franc", "spainn", "germnny"] * 4):
        df.loc[i, "Geography"] = bad
    df.loc[pick(3), "Geography"] = "usa"
    df.loc[pick(5), "Balance"] = -99999.99
    df.loc[pick(3), "Balance"] = 999999999.99
    df.loc[pick(4), "EstimatedSalary"] = 99999999.99
    for i, bad in zip(pick(6), [-1, 0, 80, 999, 80, 0]):
        df.loc[i, "NumOfProducts"] = bad
    df.loc[pick(4), "HasCrCard"] = 2
    df.loc[pick(4), "CreditScore"] = 999
    df.loc[pick(3), "Age"] = np.nan
    dupes = df.sample(15, random_state=1)
    return pd.concat([df, dupes], ignore_index=True)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    dirty = make_dirty(rng)
    dirty.to_csv(OUT / "churn_sample_dirty.csv", index=False)
    unl = make_clean(500, rng, labelled=False).drop(columns=["RowNumber"])
    unl.to_csv(OUT / "churn_sample_unlabeled.csv", index=False)
    print(f"dirty: {len(dirty)} rows | unlabeled: {len(unl)} rows -> {OUT}")
