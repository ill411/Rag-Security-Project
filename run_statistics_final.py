"""
Final Statistical Analysis

Runs McNemar's test with contingency tables across every experimental
condition generated at temperature 0:

  Part A  ablation: each LEDA configuration vs baseline, both pipelines
  Part B  component contribution: marginal effect of each component
  Part C  held-out: baseline vs LEDA on unseen prompts
  Part D  generalisation: classifier flag rate, original vs held-out

Uses the chi-square survival function with one degree of freedom and
reports the exact binomial test where discordant counts are small.
"""

import csv
import math
from pathlib import Path
from collections import defaultdict
from scipy.stats import chi2, binomtest

ABLATION = Path("results/ablation/ablation_detail.csv")
HELDOUT = Path("results/heldout/heldout_detail.csv")

CONDITION_LABELS = {
    "C0": "baseline",
    "C1": "intent only",
    "C2": "wrapper only",
    "C3": "guard only",
    "C4": "intent + wrapper",
    "C5": "intent + guard",
    "C6": "wrapper + guard",
    "C7": "full LEDA",
}


def binarise(score):
    return 1 if int(score) >= 1 else 0


def load_ablation():
    """Returns {pipeline: {condition: {prompt_id: binary_score}}}"""
    data = defaultdict(lambda: defaultdict(dict))
    for r in csv.DictReader(open(ABLATION, encoding="utf-8")):
        if r["category"] == "benign":
            continue
        data[r["pipeline"]][r["condition"]][r["prompt_id"]] = binarise(r["score"])
    return data


def load_ablation_benign():
    """Returns {pipeline: {condition: [scores]}}"""
    data = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(ABLATION, encoding="utf-8")):
        if r["category"] != "benign":
            continue
        data[r["pipeline"]][r["condition"]].append(int(r["score"]))
    return data


def load_heldout():
    data = defaultdict(lambda: defaultdict(dict))
    for r in csv.DictReader(open(HELDOUT, encoding="utf-8")):
        data[r["pipeline"]][r["condition"]][r["prompt_id"]] = binarise(r["score"])
    return data


def paired(a_dict, b_dict):
    """Align two condition dicts by prompt_id, return paired lists."""
    ids = sorted(set(a_dict) & set(b_dict))
    return [a_dict[i] for i in ids], [b_dict[i] for i in ids]


def contingency(a, b):
    n11 = sum(1 for x, y in zip(a, b) if x == 1 and y == 1)
    n10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n00 = sum(1 for x, y in zip(a, b) if x == 0 and y == 0)
    return n11, n10, n01, n00


def mcnemar(n10, n01):
    disc = n10 + n01
    if disc == 0:
        return 0.0, 1.0, 1.0
    stat = (abs(n10 - n01) - 1) ** 2 / disc
    p_chi = chi2.sf(stat, df=1)
    p_exact = binomtest(n10, n=disc, p=0.5, alternative="two-sided").pvalue
    return round(stat, 3), p_chi, p_exact


def cohen_h(p1, p2):
    return round(
        abs(2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))), 3
    )


def effect_label(h):
    if h >= 0.8:
        return "large"
    if h >= 0.5:
        return "medium"
    if h >= 0.2:
        return "small"
    return "negligible"


def fmt_p(p):
    return "< 0.001" if p < 0.001 else f"{p:.4f}"


def compare(label_a, a, label_b, b, verbose=True):
    n = len(a)
    rate_a = sum(a) / n * 100
    rate_b = sum(b) / n * 100
    n11, n10, n01, n00 = contingency(a, b)
    disc = n10 + n01
    stat, p_chi, p_exact = mcnemar(n10, n01)
    h = cohen_h(rate_a / 100, rate_b / 100)
    p_use = p_exact if disc < 25 else p_chi
    sig = p_use < 0.05

    if verbose:
        print(f"\n  {label_a}  ->  {label_b}")
        print(f"    rate: {rate_a:.1f}%  ->  {rate_b:.1f}%   "
              f"(delta {rate_a - rate_b:+.1f} pp)")
        print(f"    contingency: n11={n11} n10={n10} n01={n01} n00={n00}  "
              f"(discordant {disc})")
        print(f"    chi2={stat}  p_chi={fmt_p(p_chi)}  "
              f"p_exact={fmt_p(p_exact)}")
        print(f"    Cohen's h={h} ({effect_label(h)})   "
              f"{'SIGNIFICANT' if sig else 'not significant'}")

    return {
        "rate_a": round(rate_a, 1),
        "rate_b": round(rate_b, 1),
        "n10": n10,
        "n01": n01,
        "disc": disc,
        "chi2": stat,
        "p": p_use,
        "h": h,
        "sig": sig,
    }


def part_a(abl):
    print("\n" + "=" * 68)
    print("PART A — ABLATION: EACH CONFIGURATION VS BASELINE")
    print("=" * 68)

    results = {}
    for pipe in abl:
        print(f"\n{pipe}")
        print("-" * 68)
        base = abl[pipe]["C0"]
        results[pipe] = {}
        for cond in ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]:
            if cond not in abl[pipe]:
                continue
            a, b = paired(base, abl[pipe][cond])
            r = compare("baseline", a, CONDITION_LABELS[cond], b)
            results[pipe][cond] = r
    return results


def part_b(abl):
    print("\n" + "=" * 68)
    print("PART B — MARGINAL COMPONENT CONTRIBUTION")
    print("=" * 68)
    print("Each pair isolates one component by holding the others fixed.")

    # (without, with, component being added)
    pairs = [
        ("C0", "C1", "intent"),
        ("C2", "C4", "intent"),
        ("C3", "C5", "intent"),
        ("C6", "C7", "intent"),
        ("C0", "C2", "wrapper"),
        ("C1", "C4", "wrapper"),
        ("C3", "C6", "wrapper"),
        ("C5", "C7", "wrapper"),
        ("C0", "C3", "guard"),
        ("C1", "C5", "guard"),
        ("C2", "C6", "guard"),
        ("C4", "C7", "guard"),
    ]

    for pipe in abl:
        print(f"\n{pipe}")
        print("-" * 68)
        by_comp = defaultdict(list)
        for without, with_, comp in pairs:
            if without not in abl[pipe] or with_ not in abl[pipe]:
                continue
            a, b = paired(abl[pipe][without], abl[pipe][with_])
            rate_a = sum(a) / len(a) * 100
            rate_b = sum(b) / len(b) * 100
            by_comp[comp].append(rate_a - rate_b)

        print(f"  {'Component':<12} {'Mean reduction':>16} "
              f"{'Range':>22}")
        for comp in ["intent", "wrapper", "guard"]:
            d = by_comp[comp]
            if not d:
                continue
            mean = sum(d) / len(d)
            print(f"  {comp:<12} {mean:>15.1f} pp "
                  f"{min(d):>10.1f} to {max(d):<8.1f}")


def part_b_utility(benign):
    print("\n" + "=" * 68)
    print("PART B2 — UTILITY COST BY COMPONENT")
    print("=" * 68)

    pairs = [
        ("C0", "C1", "intent"),
        ("C2", "C4", "intent"),
        ("C3", "C5", "intent"),
        ("C6", "C7", "intent"),
        ("C0", "C2", "wrapper"),
        ("C1", "C4", "wrapper"),
        ("C3", "C6", "wrapper"),
        ("C5", "C7", "wrapper"),
        ("C0", "C3", "guard"),
        ("C1", "C5", "guard"),
        ("C2", "C6", "guard"),
        ("C4", "C7", "guard"),
    ]

    for pipe in benign:
        print(f"\n{pipe}")
        print("-" * 68)
        by_comp = defaultdict(list)
        for without, with_, comp in pairs:
            if without not in benign[pipe] or with_ not in benign[pipe]:
                continue
            a = benign[pipe][without]
            b = benign[pipe][with_]
            urs_a = sum(a) / len(a) * 100
            urs_b = sum(b) / len(b) * 100
            by_comp[comp].append(urs_a - urs_b)

        print(f"  {'Component':<12} {'Mean URS cost':>16}")
        for comp in ["intent", "wrapper", "guard"]:
            d = by_comp[comp]
            if not d:
                continue
            mean = sum(d) / len(d)
            print(f"  {comp:<12} {mean:>15.1f} pp")


def part_c(held):
    print("\n" + "=" * 68)
    print("PART C — HELD-OUT SET: BASELINE VS FULL LEDA")
    print("=" * 68)

    for pipe in held:
        print(f"\n{pipe}")
        print("-" * 68)
        a, b = paired(held[pipe]["baseline"], held[pipe]["LEDA"])
        compare("baseline", a, "LEDA", b)


def part_d():
    print("\n" + "=" * 68)
    print("PART D — CLASSIFIER GENERALISATION")
    print("=" * 68)

    import re
    import sys
    sys.path.insert(0, ".")
    from defences.leda import classify_intent

    for path, label in [
        ("prompts/adversarial_prompts.csv", "Original (design set)"),
        ("prompts/heldout_prompts.csv", "Held-out (unseen)"),
    ]:
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        flagged = sum(1 for r in rows if classify_intent(r["prompt_text"])[0])
        print(f"\n  {label}")
        print(f"    flagged {flagged}/{len(rows)} "
              f"({flagged/len(rows)*100:.1f}%)")

    print("\n  A large gap indicates the classifier matched vocabulary")
    print("  specific to the design set rather than extraction intent.")


def main():
    print("FINAL STATISTICAL ANALYSIS")
    print("All conditions, temperature 0, fixed seed")

    abl = load_ablation()
    benign = load_ablation_benign()
    held = load_heldout()

    part_a(abl)
    part_b(abl)
    part_b_utility(benign)
    part_c(held)
    part_d()

    print("\n" + "=" * 68)
    print("Reporting guidance")
    print("=" * 68)
    print("  Cite p_exact where discordant n < 25")
    print("  Report n10 and n01 alongside every test")
    print("  Part B isolates component effects; Part A tests configurations")


if __name__ == "__main__":
    main()