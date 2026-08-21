"""
Statistical Analysis — McNemar's Test
Compares baseline vs LEDA attack results for both pipelines.
Reports McNemar's test statistic, p-value, and Cohen's h effect size.
"""

import csv
import math
from pathlib import Path


def load_scores(filepath):
    return [int(r["score"]) for r in csv.DictReader(open(filepath, encoding="utf-8"))]


def binary(scores):
    return [1 if s >= 1 else 0 for s in scores]


def mcnemar(baseline, leda):
    n01 = sum(1 for b, l in zip(baseline, leda) if b == 1 and l == 0)
    n10 = sum(1 for b, l in zip(baseline, leda) if b == 0 and l == 1)
    if n01 + n10 == 0:
        return 0.0, 1.0
    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p = math.exp(-chi2 / 2)
    return round(chi2, 2), round(p, 4)


def cohen_h(p1, p2):
    h = 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))
    return round(abs(h), 3)


def asr(scores):
    b = binary(scores)
    return round(sum(b) / len(b) * 100, 1)


def analyse(pipeline, baseline_path, leda_path, threshold=12):
    print(f"\n{'='*55}")
    print(f"{pipeline} — Baseline vs LEDA (threshold {threshold})")
    print(f"{'='*55}")

    baseline = load_scores(baseline_path)
    leda = load_scores(leda_path)

    base_asr = asr(baseline)
    leda_dbr = asr(leda)

    print(f"Baseline ASR:     {base_asr}%")
    print(f"LEDA DBR:         {leda_dbr}%")
    print(f"Reduction:        {round(base_asr - leda_dbr, 1)} percentage points")

    chi2, p = mcnemar(binary(baseline), binary(leda))
    h = cohen_h(base_asr / 100, leda_dbr / 100)

    print(f"\nMcNemar's test:   chi2={chi2}, p={p}")
    print(f"Cohen's h:        {h}")

    if p < 0.05:
        print("Result:           SIGNIFICANT (p < 0.05)")
    else:
        print("Result:           NOT SIGNIFICANT (p >= 0.05)")

    if h >= 0.8:
        magnitude = "large"
    elif h >= 0.5:
        magnitude = "medium"
    elif h >= 0.2:
        magnitude = "small"
    else:
        magnitude = "negligible"
    print(f"Effect size:      {magnitude}")

    print("\nBy attack category:")
    cats = {}
    for path, label in [(baseline_path, "baseline"), (leda_path, "leda")]:
        for row in csv.DictReader(open(path, encoding="utf-8")):
            cat = row["category"]
            cats.setdefault(cat, {}).setdefault(label, []).append(int(row["score"]))

    for cat in ["direct", "indirect", "iterative"]:
        b_asr = asr(cats[cat]["baseline"])
        l_dbr = asr(cats[cat]["leda"])
        print(f"  {cat}: {b_asr}% -> {l_dbr}% (delta {round(b_asr - l_dbr, 1)}pp)")


def main():
    print("Statistical Analysis: Baseline vs LEDA")
    print("McNemar's test (paired binary outcomes)")

    analyse(
        "Llama 3.1 8B",
        "results/llama_baseline_attack.csv",
        "results/llama_leda_t12.csv",
    )

    analyse(
        "GPT-4o mini",
        "results/gpt4o_baseline_attack.csv",
        "results/gpt4o_leda_t12.csv",
    )


if __name__ == "__main__":
    main()