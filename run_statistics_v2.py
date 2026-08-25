"""
Corrected Statistical Analysis — McNemar's Test

Fixes two problems in the original implementation:
1. The p-value used an incorrect approximation rather than the
   chi-square survival function with 1 degree of freedom.
2. Discordant cell counts were not reported, making the test
   result impossible to verify independently.

Reports the full 2x2 paired contingency table, McNemar's statistic
with continuity correction, the exact binomial test as a check for
small discordant counts, and Cohen's h effect size.
"""

import csv
import math
from scipy.stats import chi2, binomtest


def load_scores(filepath):
    return [int(r["score"]) for r in csv.DictReader(open(filepath, encoding="utf-8"))]


def binarise(scores):
    """Convert 0/1/2 extraction scores to binary success/failure."""
    return [1 if s >= 1 else 0 for s in scores]


def contingency(baseline, defended):
    """
    Build the paired 2x2 table.
    n11: succeeded both baseline and defended
    n10: succeeded baseline, blocked by defence  (defence helped)
    n01: blocked at baseline, succeeded defended (defence hurt)
    n00: failed both
    """
    n11 = sum(1 for b, d in zip(baseline, defended) if b == 1 and d == 1)
    n10 = sum(1 for b, d in zip(baseline, defended) if b == 1 and d == 0)
    n01 = sum(1 for b, d in zip(baseline, defended) if b == 0 and d == 1)
    n00 = sum(1 for b, d in zip(baseline, defended) if b == 0 and d == 0)
    return n11, n10, n01, n00


def mcnemar(n10, n01, correction=True):
    """
    McNemar's test on the discordant cells only.
    Uses continuity correction by default (Edwards correction).
    Returns (statistic, p_value) using chi-square with df=1.
    """
    disc = n10 + n01
    if disc == 0:
        return 0.0, 1.0
    if correction:
        stat = (abs(n10 - n01) - 1) ** 2 / disc
    else:
        stat = (n10 - n01) ** 2 / disc
    p = chi2.sf(stat, df=1)
    return round(stat, 3), p


def exact_mcnemar(n10, n01):
    """
    Exact binomial test on discordant pairs.
    Preferred when discordant count is small (< 25).
    """
    disc = n10 + n01
    if disc == 0:
        return 1.0
    result = binomtest(n10, n=disc, p=0.5, alternative="two-sided")
    return result.pvalue


def cohen_h(p1, p2):
    return round(abs(2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))), 3)


def effect_label(h):
    if h >= 0.8:
        return "large"
    if h >= 0.5:
        return "medium"
    if h >= 0.2:
        return "small"
    return "negligible"


def fmt_p(p):
    if p < 0.001:
        return "< 0.001"
    return f"{p:.4f}"


def analyse(name, baseline_path, defended_path, condition_label):
    base = binarise(load_scores(baseline_path))
    dfnd = binarise(load_scores(defended_path))

    n = len(base)
    asr = sum(base) / n * 100
    dbr = sum(dfnd) / n * 100

    n11, n10, n01, n00 = contingency(base, dfnd)
    disc = n10 + n01

    stat, p_chi = mcnemar(n10, n01, correction=True)
    p_exact = exact_mcnemar(n10, n01)
    h = cohen_h(asr / 100, dbr / 100)

    print()
    print("=" * 62)
    print(f"{name} — Baseline vs {condition_label}")
    print("=" * 62)
    print(f"Prompts evaluated:        {n}")
    print(f"Baseline ASR:             {asr:.1f}%")
    print(f"Defended DBR:             {dbr:.1f}%")
    print(f"Absolute reduction:       {asr - dbr:.1f} percentage points")
    print()
    print("Paired 2x2 contingency table")
    print("-" * 62)
    print(f"  Succeeded both conditions          (n11): {n11}")
    print(f"  Baseline success, defence blocked  (n10): {n10}   <- discordant")
    print(f"  Baseline blocked, defence success  (n01): {n01}   <- discordant")
    print(f"  Failed both conditions             (n00): {n00}")
    print(f"  Total discordant pairs:                   {disc}")
    print()
    print("McNemar's test (df = 1, continuity corrected)")
    print("-" * 62)
    print(f"  chi-square statistic:  {stat}")
    print(f"  p-value (chi-square):  {fmt_p(p_chi)}")
    print(f"  p-value (exact binom): {fmt_p(p_exact)}")
    if disc < 25:
        print(f"  NOTE: discordant n = {disc} < 25, exact test preferred")
    print()
    print("Effect size")
    print("-" * 62)
    print(f"  Cohen's h:             {h}  ({effect_label(h)})")
    print()
    verdict_p = p_exact if disc < 25 else p_chi
    if verdict_p < 0.05:
        print(f"  RESULT: SIGNIFICANT (p = {fmt_p(verdict_p)} < 0.05)")
    else:
        print(f"  RESULT: NOT SIGNIFICANT (p = {fmt_p(verdict_p)} >= 0.05)")


def by_category(name, baseline_path, defended_path):
    print()
    print(f"Category breakdown — {name}")
    print("-" * 62)
    cats = {}
    for path, label in [(baseline_path, "base"), (defended_path, "dfnd")]:
        for row in csv.DictReader(open(path, encoding="utf-8")):
            cats.setdefault(row["category"], {}).setdefault(label, []).append(
                1 if int(row["score"]) >= 1 else 0
            )
    for cat in ["direct", "indirect", "iterative"]:
        if cat not in cats:
            continue
        b = cats[cat]["base"]
        d = cats[cat]["dfnd"]
        b_rate = sum(b) / len(b) * 100
        d_rate = sum(d) / len(d) * 100
        n11, n10, n01, n00 = contingency(b, d)
        print(
            f"  {cat:<10} {b_rate:5.1f}% -> {d_rate:5.1f}%  "
            f"(n10={n10}, n01={n01}, n={len(b)})"
        )


def main():
    print("CORRECTED STATISTICAL ANALYSIS")
    print("McNemar's test with contingency tables and exact binomial check")

    configs = [
        (
            "Llama 3.1 8B",
            "results/llama_baseline_attack.csv",
            "results/llama_leda_t12.csv",
            "LEDA (threshold 12)",
        ),
        (
            "GPT-4o mini",
            "results/gpt4o_baseline_attack.csv",
            "results/gpt4o_leda_t12.csv",
            "LEDA (threshold 12)",
        ),
        (
            "GPT-4o mini",
            "results/gpt4o_baseline_attack.csv",
            "results/gpt4o_leda_judge_attack.csv",
            "LEDA-J (LLM judge)",
        ),
    ]

    for name, base, dfnd, label in configs:
        try:
            analyse(name, base, dfnd, label)
            by_category(f"{name} / {label}", base, dfnd)
        except FileNotFoundError as e:
            print(f"\nSKIPPED {name} / {label}: {e}")

    print()
    print("=" * 62)
    print("Interpretation notes for write-up:")
    print("  - Report n10 and n01 in the thesis; they are what the test uses")
    print("  - Where discordant n < 25, cite the exact binomial p-value")
    print("  - Cohen's h describes magnitude, p describes reliability")
    print("=" * 62)


if __name__ == "__main__":
    main()