import os
import json
import pandas as pd
import numpy as np
import scipy.stats as st

INPUT_FILE = "phase37_final_dok_scored_32b.jsonl"
REPORT_FILE = "phase37_dok_significance_report.txt"

PIPELINES = {
    "Baseline_Closed": "Baseline (Closed-Book)",
    "Baseline_Open": "Baseline (Open-Book Optimal)",
    "SFT_Closed": "Phase 37 SFT (Closed-Book)",
    "SFT_Open": "Phase 37 SFT (Open-Book Optimal)"
}

TARGET_MAP = {
    "foundational": "Level 1",
    "intermediate": "Level 2",
    "advanced": "Level 3"
}

def calc_stats(data_list):
    arr = np.array(data_list)
    if len(arr) < 2: return np.mean(arr), 0.0, 0.0
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    # 95% Confidence Interval
    ci_margin = st.t.ppf(0.975, len(arr)-1) * st.sem(arr) if len(arr) > 1 else 0.0
    return mean, std, ci_margin

def format_line(name, data):
    m, sd, ci = calc_stats(data)
    return f"{name:<35} : {m:>5.1f}% ± {ci:>4.1f}  (SD: {sd:>5.1f})"

def get_p_value(series_A, series_B):
    df_pair = pd.concat([series_A, series_B], axis=1).dropna()
    a = df_pair.iloc[:, 0].tolist()
    b = df_pair.iloc[:, 1].tolist()
    if len(a) < 2 or a == b: return 1.0
    try:
        # Wilcoxon signed-rank test for paired non-parametric data
        _, p_val = st.wilcoxon(a, b, zero_method='wilcox')
        return p_val
    except: return 1.0

def format_p(p):
    if p < 0.001: return f"{p:.4f} (***)"
    if p < 0.01:  return f"{p:.4f} (**)"
    if p < 0.05:  return f"{p:.4f} (*)"
    return f"{p:.4f} (ns)"

def main():
    print(f"📥 Loading scored DOK data from {INPUT_FILE}...")
    flat_data = []
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            
            seed = d["seed"]
            topic = d["topic"]
            pipeline = d["pipeline"]
            tier = d["target_tier"]
            pred = d.get("eval", {}).get("predicted_level", "Unknown")
            
            expected = TARGET_MAP[tier]
            is_match = 100.0 if pred == expected else 0.0
            is_level_3 = 100.0 if pred == "Level 3" else 0.0
            is_level_1 = 100.0 if pred == "Level 1" else 0.0

            flat_data.append({
                "pair_id": f"{seed}_{topic}_{tier}", # Unique ID for paired Wilcoxon testing
                "pair_id_adv": f"{seed}_{topic}",
                "pipeline": pipeline,
                "run_seed": seed,
                "target_tier": tier,
                "match_score": is_match,
                "level3_score": is_level_3,
                "level1_score": is_level_1
            })

    df = pd.DataFrame(flat_data)
    if df.empty:
        print("❌ No data found!")
        return

    # To calculate CIs properly, we group by run_seed (or seed_topic) to get the variance. 
    # Grouping by seed to match previous script methodology
    seed_tier_means = df.groupby(['pipeline', 'run_seed', 'target_tier']).mean(numeric_only=True).reset_index()

    report = "=========================================================\n"
    report += "   CALIBRATED DOK MATCH & STATISTICAL SIGNIFICANCE \n"
    report += "=========================================================\n\n"

    # 1. DOK MATCH RATES BY TIER
    report += "--- 1. OVERALL DOK MATCH RATE (%) BY TIER ---\n"
    for tier in ["foundational", "intermediate", "advanced"]:
        report += f">> TARGET TIER: {tier.upper()} (Expected: {TARGET_MAP[tier]})\n"
        for p_key, p_name in PIPELINES.items():
            vals = seed_tier_means[(seed_tier_means['pipeline'] == p_key) & (seed_tier_means['target_tier'] == tier)]['match_score'].tolist()
            report += format_line(p_name, vals) + "\n"
        report += "\n"

    # 2. STATISTICAL SIGNIFICANCE
    report += "=========================================================\n"
    report += "   STATISTICAL SIGNIFICANCE REPORT (WILCOXON P-VALUES) \n"
    report += "=========================================================\n"
    report += "* Significance Levels: (*) p < 0.05 | (**) p < 0.01 | (***) p < 0.001 | (ns) Not Significant\n\n"
    
    # We pivot on the individual flashcard pairs to get strict paired p-values (N = Total topics * seeds)
    pivot_match = df.pivot(index='pair_id', columns='pipeline', values='match_score')
    
    # Isolate specific tiers for targeted p-values
    df_adv = df[df['target_tier'] == 'advanced']
    pivot_adv_l3 = df_adv.pivot(index='pair_id_adv', columns='pipeline', values='level3_score')
    
    df_fnd = df[df['target_tier'] == 'foundational']
    pivot_fnd_l1 = df_fnd.pivot(index='pair_id_adv', columns='pipeline', values='level1_score')

    comparisons = [
        ("SFT (Closed) vs Baseline (Closed)", "SFT_Closed", "Baseline_Closed"),
        ("SFT (Open) vs Baseline (Open)",     "SFT_Open",   "Baseline_Open"),
        ("SFT RAG Resilience (SFT Open vs Closed)", "SFT_Open", "SFT_Closed"),
        ("Baseline RAG Degradation (Base Open vs Closed)", "Baseline_Open", "Baseline_Closed")
    ]

    for comp_name, m_a, m_b in comparisons:
        report += f"📊 COMPARISON: {comp_name}\n"
        report += "-" * 60 + "\n"
        
        # P-values
        p_match_all = get_p_value(pivot_match[m_a], pivot_match[m_b])
        p_adv_l3    = get_p_value(pivot_adv_l3[m_a], pivot_adv_l3[m_b])
        p_fnd_l1    = get_p_value(pivot_fnd_l1[m_a], pivot_fnd_l1[m_b])
        
        report += f"  • Overall DOK Match Rate (All Tiers) : p = {format_p(p_match_all)}\n"
        report += f"  • Target Advanced: Hit Level 3       : p = {format_p(p_adv_l3)}\n"
        report += f"  • Target Foundational: Hit Level 1   : p = {format_p(p_fnd_l1)}\n\n"

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"✅ Significance report saved to {REPORT_FILE}")

if __name__ == "__main__":
    main()
