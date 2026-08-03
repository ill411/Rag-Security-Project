"""
Statistical Analysis — McNemar's Test
Compares baseline vs LEDA attack results for both pipelines.
Reports: McNemar's test statistic, p-value, Cohen's h effect size.
"""

import csv
import math
from pathlib import Path

def load_scores(filepath):
    scores = []
    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores.append(int(row["score"]))
    return scores

def binary(scores):
    return [1 if s >= 1 else 0 for s in scores]

def mcnemar(baseline, leda):
    b = sum(1 for a, b in zip(baseline, leda) if a == 1 and b == 0)
    c = sum(1 for a, b in zip(baseline, leda) if a == 0 and b == 1)
    if b + c == 0:
        return 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    import math
    p = math.exp(-chi2 / 2)
    return round(chi2, 3), round(p, 4)

def cohen_h(p1, p2):
    h = 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))
    return round(abs(h), 3)

def asr(scores):
    b = binary(scores)
    return round(sum(b) / len(b) * 100, 1)

def analyse(pipeline, baseline_path, leda_path):
    print(f"\n{'='*55}")
    print(f"{pipeline} — Baseline vs LEDA (threshold 12)")
    print(f"{'='*55}")

    baseline = load_scores(baseline_path)
    leda = load_scores(leda_path)

    base_asr = asr(baseline)
    leda_dbr = asr(leda)

    print(f"Baseline ASR:     {base_asr}%")
    print(f"LEDA DBR:         {leda_dbr}%")
    print(f"Reduction:        {round(base_asr - leda_dbr, 1)} percentage points")

    chi2, p = mcnemar(binary(baseline), binary(leda))
    h = cohen_h(base_asr/100, leda_dbr/100)

    print(f"\nMcNemar's test:   χ²={chi2}, p={p}")
    print(f"Cohen's h:        {h}")

    if p < 0.05:
        print("Result:           SIGNIFICANT (p < 0.05)")
    else:
        print("Result:           not significant (p >= 0.05)")

    if h >= 0.5:
        magnitude = "large"
    elif h >= 0.3:
        magnitude = "medium"
    else:
        magnitude = "small"
    print(f"Effect size:      {magnitude}")

    print(f"\nBy category:")
    cats = {}
    for path, label in [(baseline_path, "baseline"), (leda_path, "leda")]:
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cat = row["category"]
                cats.setdefault(cat, {}).setdefault(label, []).append(int(row["score"]))

    for cat in ["direct", "indirect", "iterative"]:
        b_asr = asr(cats[cat]["baseline"])
        l_dbr = asr(cats[cat]["leda"])
        print(f"  {cat}: {b_asr}% → {l_dbr}% (Δ {round(b_asr-l_dbr,1)}pp)")

def main():
    print("Statistical Analysis: Baseline vs LEDA")
    print("McNemar's test (paired binary outcomes)")

    analyse(
        "Llama 3.1 8B",
        "results/llama_baseline_attack.csv",
        "results/llama_leda_t12.csv"
    )

    analyse(
        "GPT-4o mini",
        "results/gpt4o_baseline_attack.csv",
        "results/llama_leda_t12.csv"
    )

if __name__ == "__main__":
    main()