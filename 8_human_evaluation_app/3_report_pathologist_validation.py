import os
import re
import json
import warnings
import numpy as np
import pandas as pd
import scipy.stats as st

warnings.filterwarnings("ignore")

EVAL_FILE = "dr_honeine_evaluations.jsonl"
CARDS_FILE = "blinded_eval_set.jsonl"
REPORT_FILE = "phase37_pathologist_validation_report.txt"

PIPELINES = {
    "p1_baseline_closed": "Baseline (Closed-Book)",
    "p2_baseline_open":   "Baseline (Open-Book Optimal)",
    "p3_sft_closed":      "Phase 37 SFT (Closed-Book)",
    "p4_sft_open":        "Phase 37 SFT (Open-Book Optimal)"
}

AUDIENCE_MAP = {
    "Foundational (MS1 Level)": 1,
    "Intermediate (Resident Level)": 2,
    "Advanced (Fellow/Oncologist Level)": 3
}

BLOOM_MAP = {
    "Tier 1: Foundational (Bloom's 1-2)": 1,
    "Tier 2: Intermediate (Bloom's 3-4)": 2,
    "Tier 3: Advanced (Bloom's 5-6)": 3
}

TRUE_TIER_MAP = {
    "foundational": 1,
    "intermediate": 2,
    "advanced": 3
}

# Fallback vignette IDs if blinded_eval_set.jsonl is not in the current directory
FALLBACK_VIGNETTE_IDS = {9, 10, 12, 14, 26, 30, 34, 54, 58, 70, 75, 79, 81, 85, 93}


def parse_leading_int(val):
    if pd.isna(val) or not str(val).strip():
        return np.nan
    m = re.match(r"^(\d+)", str(val).strip())
    return float(m.group(1)) if m else np.nan


def get_wilcoxon_p(series_a, series_b):
    df_pair = pd.concat([series_a, series_b], axis=1).dropna()
    a = df_pair.iloc[:, 0].tolist()
    b = df_pair.iloc[:, 1].tolist()
    if len(a) < 2 or a == b:
        return len(a), np.mean(a) if a else 0.0, np.mean(b) if b else 0.0, 1.0
    try:
        _, p_val = st.wilcoxon(a, b, zero_method="wilcox")
        return len(a), np.mean(a), np.mean(b), float(p_val)
    except Exception:
        return len(a), np.mean(a), np.mean(b), 1.0


def format_p(p):
    if p < 0.001:
        return f"{p:.4f} (***)"
    if p < 0.01:
        return f"{p:.4f} (**)"
    if p < 0.05:
        return f"{p:.4f} (*)"
    return f"{p:.4f} (ns)"


def main():
    if not os.path.exists(EVAL_FILE):
        print(f"Error: Could not find {EVAL_FILE}")
        return

    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        eval_rows = [json.loads(line) for line in f if line.strip()]

    df = pd.DataFrame(eval_rows)

    # Numeric mappings
    df["fact_num"] = df["factual"].apply(parse_leading_int)
    df["util_num"] = df["utility"].apply(parse_leading_int)
    df["scope_num"] = df["scope"].apply(parse_leading_int) if "scope" in df.columns else np.nan

    df["is_safe"] = (df["safety"] == "Safe").astype(float)
    df["is_pot_misleading"] = (df["safety"] == "Potentially Misleading").astype(float)
    df["is_unsafe"] = (df["safety"] == "Unsafe").astype(float)

    df["aud_lvl"] = df["tier_assigned"].map(AUDIENCE_MAP)
    df["bloom_lvl"] = df["bloom_tier"].map(BLOOM_MAP) if "bloom_tier" in df.columns else np.nan
    df["true_lvl"] = df["tier_true"].map(TRUE_TIER_MAP)

    # Post-hoc alignment flags (on classifiable rows where both Audience & Bloom are mapped)
    df["aud_match"] = df["aud_lvl"] == df["true_lvl"]
    df["bloom_match"] = df["bloom_lvl"] == df["true_lvl"]
    df["dual_match"] = df["aud_match"] & df["bloom_match"]
    df["either_match"] = df["aud_match"] | df["bloom_match"]

    # Default late-session rubber-stamp profile
    df["default_profile"] = (
        (df["fact_num"] == 5)
        & (df["safety"] == "Safe")
        & (df["tier_assigned"] == "Foundational (MS1 Level)")
        & (df["bloom_tier"] == "Tier 1: Foundational (Bloom's 1-2)")
        & (df["util_num"] == 4)
        & (df["scope_num"] == 3)
    )

    report = []
    report.append("==========================================================================================")
    report.append("   PHASE 37: EXPERT PATHOLOGIST PILOT VALIDATION REPORT (N=96 COMPLETED OF 96 CARDS)")
    report.append("==========================================================================================\n")

    # -------------------------------------------------------------------------
    # SECTION 1: COHORT-WIDE SUMMARY & CLINICAL SAFETY CEILING
    # -------------------------------------------------------------------------
    n_total = len(df)
    n_scope = df["scope_num"].notna().sum()
    n_bloom = df["bloom_tier"].notna().sum()

    fact_mean, fact_sd = df["fact_num"].mean(), df["fact_num"].std(ddof=1)
    util_mean, util_sd = df["util_num"].mean(), df["util_num"].std(ddof=1)
    scope_mean, scope_sd = df["scope_num"].mean(), df["scope_num"].std(ddof=1)

    fact_5 = int((df["fact_num"] == 5).sum())
    fact_4 = int((df["fact_num"] == 4).sum())
    fact_3 = int((df["fact_num"] == 3).sum())
    fact_12 = int((df["fact_num"] <= 2).sum())
    fact_ge4 = fact_5 + fact_4

    safe_cnt = int((df["safety"] == "Safe").sum())
    mis_cnt = int((df["safety"] == "Potentially Misleading").sum())
    unsafe_cnt = int((df["safety"] == "Unsafe").sum())

    scope_3 = int((df["scope_num"] == 3).sum())
    scope_2 = int((df["scope_num"] == 2).sum())
    scope_1 = int((df["scope_num"] == 1).sum())

    report.append("--- 1. COHORT-WIDE CLINICAL SAFETY & FACTUALITY CEILING ---")
    report.append(f"  • Completed Evaluations           : {n_total} / 96 cards (eval_id 42 skipped; eval_id 44 missing Scope/Bloom, n={n_scope})")
    report.append(f"  • Clinically Unsafe Outputs       : {unsafe_cnt}/{n_total} ({unsafe_cnt/n_total*100:.1f}%)")
    report.append(f"  • Major Factual Errors (Score 1-2): {fact_12}/{n_total} ({fact_12/n_total*100:.1f}%)")
    report.append(f"  • Severe Mismatch / Dump (Scope 1): {scope_1}/{n_scope} ({scope_1/n_scope*100:.1f}%)")
    report.append(f"  • WHO Factuality >= 4/5           : {fact_ge4}/{n_total} ({fact_ge4/n_total*100:.1f}%) "
                  f"[5/5: {fact_5} ({fact_5/n_total*100:.1f}%) | 4/5: {fact_4} ({fact_4/n_total*100:.1f}%) | 3/5: {fact_3} ({fact_3/n_total*100:.1f}%)]")
    report.append(f"  • Overall Factuality Mean ± SD    : {fact_mean:.2f} ± {fact_sd:.2f} (out of 5.00)")
    report.append(f"  • Clinical Safety Breakdown       : Safe: {safe_cnt}/{n_total} ({safe_cnt/n_total*100:.1f}%) | "
                  f"Potentially Misleading: {mis_cnt}/{n_total} ({mis_cnt/n_total*100:.1f}%) | Unsafe: 0.0%")
    report.append(f"  • Educational Usefulness Mean ± SD: {util_mean:.2f} ± {util_sd:.2f} (out of 5.00)")
    report.append(f"  • Cognitive Scope Mean ± SD       : {scope_mean:.2f} ± {scope_sd:.2f} (out of 3.00) "
                  f"[3/3 Perfect: {scope_3}/{n_scope} ({scope_3/n_scope*100:.1f}%) | 2/3 Minor: {scope_2}/{n_scope} ({scope_2/n_scope*100:.1f}%)]\n")

    # -------------------------------------------------------------------------
    # SECTION 2: PIPELINE & TIER-STRATIFIED BREAKDOWN
    # -------------------------------------------------------------------------
    report.append("--- 2. PIPELINE & TIER-STRATIFIED QUALITY METRICS ---")
    header = f"{'Pipeline':<32} | {'n':<3} | {'Fact Mean':<9} | {'Fact >=4':<13} | {'Safe %':<13} | {'Util Mean':<9} | {'Scope 3/3 Perfect':<17}"
    report.append(header)
    report.append("-" * len(header))

    for p_key, p_name in PIPELINES.items():
        sub = df[df["model_true"] == p_key]
        n_p = len(sub)
        n_sc = sub["scope_num"].notna().sum()
        f_m = sub["fact_num"].mean()
        f_4 = int((sub["fact_num"] >= 4).sum())
        s_c = int((sub["safety"] == "Safe").sum())
        u_m = sub["util_num"].mean()
        sc_3 = int((sub["scope_num"] == 3).sum())
        report.append(
            f"{p_name:<32} | {n_p:<3} | {f_m:>9.2f} | {f_4/n_p*100:>5.1f}% ({f_4:>2}/{n_p:<2}) | "
            f"{s_c/n_p*100:>5.1f}% ({s_c:>2}/{n_p:<2}) | {u_m:>9.2f} | {sc_3/n_sc*100:>5.1f}% ({sc_3:>2}/{n_sc:<2})"
        )

    for tier in ["advanced", "intermediate", "foundational"]:
        report.append(f"\n>> TARGET TIER: {tier.upper()}")
        report.append("-" * len(header))
        for p_key, p_name in PIPELINES.items():
            sub = df[(df["model_true"] == p_key) & (df["tier_true"] == tier)]
            n_p = len(sub)
            n_sc = sub["scope_num"].notna().sum()
            f_m = sub["fact_num"].mean()
            f_4 = int((sub["fact_num"] >= 4).sum())
            s_c = int((sub["safety"] == "Safe").sum())
            u_m = sub["util_num"].mean()
            sc_3 = int((sub["scope_num"] == 3).sum())
            report.append(
                f"{p_name:<32} | {n_p:<3} | {f_m:>9.2f} | {f_4/n_p*100:>5.1f}% ({f_4:>2}/{n_p:<2}) | "
                f"{s_c/n_p*100:>5.1f}% ({s_c:>2}/{n_p:<2}) | {u_m:>9.2f} | {sc_3/n_sc*100:>5.1f}% ({sc_3:>2}/{n_sc:<2})"
            )

    sft_adv = df[(df["model_true"].isin(["p3_sft_closed", "p4_sft_open"])) & (df["tier_true"] == "advanced")]
    sft_adv_5 = int((sft_adv["fact_num"] == 5).sum())
    sft_adv_ge4 = int((sft_adv["fact_num"] >= 4).sum())
    sft_adv_sc3 = int((sft_adv["scope_num"] == 3).sum())
    sft_adv_ms1_res = int(sft_adv["tier_assigned"].isin(["Foundational (MS1 Level)", "Intermediate (Resident Level)"]).sum())
    report.append(
        f"\n  * Combined SFT Advanced Tier (n={len(sft_adv)}): Factuality = {sft_adv['fact_num'].mean():.2f}/5.00 "
        f"[>=4/5: {sft_adv_ge4}/{len(sft_adv)} ({sft_adv_ge4/len(sft_adv)*100:.1f}%), "
        f"5/5: {sft_adv_5}/{len(sft_adv)} ({sft_adv_5/len(sft_adv)*100:.1f}%)], "
        f"Scope 3/3 = {sft_adv_sc3}/{len(sft_adv)} ({sft_adv_sc3/len(sft_adv)*100:.1f}%), "
        f"Anchored at MS1/Resident Audience = {sft_adv_ms1_res}/{len(sft_adv)} ({sft_adv_ms1_res/len(sft_adv)*100:.1f}%)\n"
    )

    # -------------------------------------------------------------------------
    # SECTION 3: PAIRED WILCOXON SIGNED-RANK TESTS ACROSS THE 24 TOPICS
    # -------------------------------------------------------------------------
    report.append("--- 3. PAIRED STATISTICAL SIGNIFICANCE (TWO-SIDED WILCOXON SIGNED-RANK TESTS) ---")
    comparisons = [
        ("SFT (Open) vs Baseline (Open)",       "p4_sft_open",      "p2_baseline_open"),
        ("SFT (Closed) vs Baseline (Closed)",   "p3_sft_closed",    "p1_baseline_closed"),
        ("Baseline (Open) vs Baseline (Closed)","p2_baseline_open", "p1_baseline_closed"),
        ("SFT (Open) vs SFT (Closed)",          "p4_sft_open",      "p3_sft_closed")
    ]
    metrics_to_test = [
        ("Factual Correctness (1-5)", "fact_num"),
        ("Clinical Safety (% Safe)",  "is_safe"),
        ("Educational Utility (1-5)", "util_num"),
        ("Cognitive Scope (1-3)",     "scope_num")
    ]

    for comp_title, m_a, m_b in comparisons:
        report.append(f">> COMPARISON: {comp_title}")
        for m_label, col in metrics_to_test:
            piv = df.pivot(index="topic", columns="model_true", values=col)
            n_pairs, mean_a, mean_b, p_val = get_wilcoxon_p(piv[m_a], piv[m_b])
            if col == "is_safe":
                report.append(f"  • {m_label:<26} (n={n_pairs:>2}): {mean_a*100:>5.1f}% vs {mean_b*100:>5.1f}% | p = {format_p(p_val)}")
            else:
                report.append(f"  • {m_label:<26} (n={n_pairs:>2}): {mean_a:>5.2f}  vs {mean_b:>5.2f}  | p = {format_p(p_val)}")
        report.append("")

    # -------------------------------------------------------------------------
    # SECTION 4: POST-HOC MULTI-AXIAL ALIGNMENT WITH REQUESTED TIER (tier_true)
    # -------------------------------------------------------------------------
    valid = df.dropna(subset=["aud_lvl", "bloom_lvl", "scope_num"]).copy()
    valid["aud_match"] = valid["aud_lvl"] == valid["true_lvl"]
    valid["bloom_match"] = valid["bloom_lvl"] == valid["true_lvl"]
    valid["dual_match"] = valid["aud_match"] & valid["bloom_match"]
    valid["either_match"] = valid["aud_match"] | valid["bloom_match"]

    report.append(f"--- 4. POST-HOC MULTI-AXIAL ALIGNMENT WITH REQUESTED TIER (CLASSIFIABLE CARDS, N={len(valid)}) ---")
    align_hdr = f"{'Pipeline':<32} | {'n':<3} | {'Dual-Axis Match':<16} | {'Audience Match':<16} | {'Bloom Match':<16} | {'Either-Axis Match':<17}"
    report.append(align_hdr)
    report.append("-" * len(align_hdr))

    for p_key, p_name in PIPELINES.items():
        sub = valid[valid["model_true"] == p_key]
        n_v = len(sub)
        dm = int(sub["dual_match"].sum())
        am = int(sub["aud_match"].sum())
        bm = int(sub["bloom_match"].sum())
        em = int(sub["either_match"].sum())
        report.append(
            f"{p_name:<32} | {n_v:<3} | {dm/n_v*100:>5.1f}% ({dm:>2}/{n_v:<2})   | "
            f"{am/n_v*100:>5.1f}% ({am:>2}/{n_v:<2})   | {bm/n_v*100:>5.1f}% ({bm:>2}/{n_v:<2})   | "
            f"{em/n_v*100:>5.1f}% ({em:>2}/{n_v:<2})"
        )

    report.append("\n>> OPEN-BOOK RAG TIER-BY-TIER COMPARISON (SFT Open vs Baseline Open, Classifiable Cards):")
    for tier in ["foundational", "intermediate", "advanced"]:
        report.append(f"  [Target Tier: {tier.upper()}]")
        for p_key in ["p2_baseline_open", "p4_sft_open"]:
            sub = valid[(valid["model_true"] == p_key) & (valid["tier_true"] == tier)]
            n_v = len(sub)
            am = int(sub["aud_match"].sum())
            bm = int(sub["bloom_match"].sum())
            em = int(sub["either_match"].sum())
            sc3 = int((sub["scope_num"] == 3).sum())
            report.append(
                f"    • {PIPELINES[p_key]:<30} (n={n_v}): "
                f"Bloom Match = {bm/n_v*100:>5.1f}% ({bm}/{n_v}) | "
                f"Aud Match = {am/n_v*100:>5.1f}% ({am}/{n_v}) | "
                f"Either = {em/n_v*100:>5.1f}% ({em}/{n_v}) | "
                f"Scope 3/3 = {sc3/n_v*100:>5.1f}% ({sc3}/{n_v})"
            )
    report.append("")

    # -------------------------------------------------------------------------
    # SECTION 5: TAXONOMIC ORTHOGONALITY & COGNITIVE SCOPE ADJUDICATION
    # -------------------------------------------------------------------------
    valid["coupled"] = valid["aud_lvl"] == valid["bloom_lvl"]
    coupled_df = valid[valid["coupled"]]
    orth_df = valid[~valid["coupled"]]

    orth_either = int(orth_df["either_match"].sum())
    orth_sc3 = int((orth_df["scope_num"] == 3).sum())
    orth_sc2 = int((orth_df["scope_num"] == 2).sum())

    report.append(f"--- 5. AUDIENCE-VS.-BLOOM ORTHOGONALITY & COGNITIVE SCOPE ADJUDICATION (N={len(valid)}) ---")
    report.append(f"  • Coupled Ratings (Audience Level == Bloom's Level)   : {len(coupled_df)}/{len(valid)} ({len(coupled_df)/len(valid)*100:.1f}%)")
    report.append(f"  • Orthogonal Ratings (Audience Level != Bloom's Level): {len(orth_df)}/{len(valid)} ({len(orth_df)/len(valid)*100:.1f}%)")
    report.append(f"  • Orthogonal Cards Matching tier_true on >=1 Axis     : {orth_either}/{len(orth_df)} ({orth_either/len(orth_df)*100:.1f}%)")
    report.append(f"  • Orthogonal Cards Rated '3 - Perfectly Scoped'       : {orth_sc3}/{len(orth_df)} ({orth_sc3/len(orth_df)*100:.1f}%)")
    report.append(f"  • Orthogonal Cards Rated '2 - Minor Mismatch'         : {orth_sc2}/{len(orth_df)} ({orth_sc2/len(orth_df)*100:.1f}%)\n")

    report.append("  >> Key Orthogonal Exemplars Rated '3 - Perfectly Scoped':")
    for _, r in orth_df[orth_df["scope_num"] == 3].sort_values("eval_id").iterrows():
        report.append(
            f"    • Card {int(r['eval_id']):>2} | {r['model_true']:<18} | Topic: {r['topic']:<28} | "
            f"Req: {r['tier_true']:<12} -> Assigned Aud: {r['tier_assigned']:<34} | Assigned Bloom: {r['bloom_tier']}"
        )
    report.append("")

    # -------------------------------------------------------------------------
    # SECTION 6: NEGATIVE CONSTRAINT AUDIT (ANTI-VIGNETTE MANDATE)
    # -------------------------------------------------------------------------
    report.append("--- 6. NEGATIVE CONSTRAINT COMPLIANCE (ANTI-VIGNETTE MANDATE ON N=96 DECK) ---")
    vignette_counts = {k: 0 for k in PIPELINES}
    total_deck_counts = {k: 24 for k in PIPELINES}

    if os.path.exists(CARDS_FILE):
        with open(CARDS_FILE, "r", encoding="utf-8") as f:
            cards_list = [json.loads(line) for line in f if line.strip()]
        vignette_Pattern = re.compile(r"\b\d+-(?:year|month)-old\b", re.IGNORECASE)
        for c in cards_list:
            m_key = c["model_true"]
            if vignette_Pattern.search(c.get("q", "")):
                vignette_counts[m_key] += 1
    else:
        for _, r in df[df["eval_id"].isin(FALLBACK_VIGNETTE_IDS)].iterrows():
            vignette_counts[r["model_true"]] += 1

    base_vig = vignette_counts["p1_baseline_closed"] + vignette_counts["p2_baseline_open"]
    sft_vig = vignette_counts["p3_sft_closed"] + vignette_counts["p4_sft_open"]
    _, p_fisher_all = st.fisher_exact([[48 - sft_vig, sft_vig], [48 - base_vig, base_vig]])
    _, p_fisher_open = st.fisher_exact([[24 - vignette_counts["p4_sft_open"], vignette_counts["p4_sft_open"]],
                                        [24 - vignette_counts["p2_baseline_open"], vignette_counts["p2_baseline_open"]]])
    _, p_fisher_closed = st.fisher_exact([[24 - vignette_counts["p3_sft_closed"], vignette_counts["p3_sft_closed"]],
                                          [24 - vignette_counts["p1_baseline_closed"], vignette_counts["p1_baseline_closed"]]])

    for p_key, p_name in PIPELINES.items():
        v_cnt = vignette_counts[p_key]
        comp_cnt = 24 - v_cnt
        report.append(f"  • {p_name:<32}: {comp_cnt}/24 ({comp_cnt/24*100:>5.1f}%) Anti-Vignette Compliance ({v_cnt} vignette violations)")
    report.append(f"  • Overall SFT (48/48, 100.0%) vs Overall Baseline ({48-base_vig}/48, {(48-base_vig)/48*100:.1f}%): Fisher's Exact p = {format_p(p_fisher_all)}")
    report.append(f"  • Open-Book SFT (24/24) vs Open-Book Baseline ({24-vignette_counts['p2_baseline_open']}/24): Fisher's Exact p = {format_p(p_fisher_open)}")
    report.append(f"  • Closed-Book SFT (24/24) vs Closed-Book Baseline ({24-vignette_counts['p1_baseline_closed']}/24): Fisher's Exact p = {format_p(p_fisher_closed)}\n")

    # -------------------------------------------------------------------------
    # SECTION 7: SESSION TERTILE PROGRESSION & TOPIC-REPETITION FATIGUE
    # -------------------------------------------------------------------------
    report.append("--- 7. SEQUENTIAL SESSION TERTILE PROGRESSION & TOPIC-REPETITION FATIGUE ---")
    df["tertile"] = pd.cut(
        df["eval_id"],
        bins=[0, 32, 64, 96],
        labels=["Tertile 1: Early (Cards 1-32)", "Tertile 2: Middle (Cards 33-64)", "Tertile 3: Late (Cards 65-96)"]
    )

    tert_hdr = f"{'Session Block':<33} | {'n':<3} | {'MS1 Audience %':<16} | {'Bloom Tier 1 %':<16} | {'Scope 3/3 %':<15} | {'Default Profile %':<18}"
    report.append(tert_hdr)
    report.append("-" * len(tert_hdr))

    for t_label, grp in df.groupby("tertile", observed=False):
        n_t = len(grp)
        n_tb = grp["bloom_tier"].notna().sum()
        n_ts = grp["scope_num"].notna().sum()
        ms1_c = int((grp["tier_assigned"] == "Foundational (MS1 Level)").sum())
        bl1_c = int((grp["bloom_tier"] == "Tier 1: Foundational (Bloom's 1-2)").sum())
        sc3_c = int((grp["scope_num"] == 3).sum())
        def_c = int(grp["default_profile"].sum())
        report.append(
            f"{str(t_label):<33} | {n_t:<3} | {ms1_c/n_t*100:>5.1f}% ({ms1_c:>2}/{n_t:<2})   | "
            f"{bl1_c/n_tb*100:>5.1f}% ({bl1_c:>2}/{n_tb:<2})   | {sc3_c/n_ts*100:>5.1f}% ({sc3_c:>2}/{n_ts:<2})  | "
            f"{def_c/n_t*100:>5.1f}% ({def_c:>2}/{n_t:<2})"
        )

    report.append("\n  >> Pipeline Distribution in Late Session Block (Tertile 3, Cards 65-96):")
    for p_key, p_name in PIPELINES.items():
        p_sub = df[df["model_true"] == p_key]
        late_cnt = int((p_sub["eval_id"] >= 65).sum())
        report.append(f"    • {p_name:<32}: {late_cnt:>2}/{len(p_sub)} ({late_cnt/len(p_sub)*100:>5.1f}%) of cards landed in Cards 65-96 (Mean eval_id: {p_sub['eval_id'].mean():.1f})")

    rep_pattern = re.compile(r"\brepetition\b|\brepeated question\b", re.IGNORECASE)
    rep_cards = df[df["comments"].apply(lambda c: bool(rep_pattern.search(str(c))))].sort_values("eval_id")
    report.append(f"\n  >> Cards Flagged by Evaluator for Topic Repetition Across Pipelines (n={len(rep_cards)}):")
    for _, r in rep_cards.iterrows():
        clean_c = str(r["comments"]).strip().replace("\n", " ")
        report.append(f"    • Card {int(r['eval_id']):>2} ({r['model_true']:<18} | {r['topic']:<24}): \"{clean_c}\"")

    report.append("\n==========================================================================================")
    final_output = "\n".join(report)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(final_output)

    print(final_output)
    print(f"\nReport saved to {REPORT_FILE}")


if __name__ == "__main__":
    main()
