import os
import json
import pandas as pd
import numpy as np
import scipy.stats as st

CACHE_STEER = "phase37_eval_cache_steerability_4way.jsonl"
CACHE_FACT = "phase37_eval_cache_fact_density_4way.jsonl"
REPORT_FILE = "phase37_bloom_and_significance_report.txt"

PIPELINES = {
    "p1_baseline_closed": "Baseline (Closed-Book)",
    "p2_baseline_open": "Baseline (Open-Book Optimal)",
    "p3_sft_closed": "Phase 37 SFT (Closed-Book)",
    "p4_sft_open": "Phase 37 SFT (Open-Book Optimal)"
}

def calc_stats(data_list):
    arr = np.array(data_list)
    if len(arr) < 2: return np.mean(arr), 0.0, 0.0
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    ci_margin = st.t.ppf(0.975, len(arr)-1) * st.sem(arr)
    return mean, std, ci_margin

def format_line(name, data):
    m, sd, ci = calc_stats(data)
    return f"{name:<35} : {m:>4.2f} ± {ci:>4.2f}  (SD: {sd:>4.2f})"

def calculate_bce(tier, bloom_score):
    if tier == "foundational": min_b, max_b = 1.0, 2.5
    elif tier == "intermediate": min_b, max_b = 3.0, 4.0
    elif tier == "advanced": min_b, max_b = 4.5, 6.0
    else: return 0.0
    if bloom_score < min_b: return min_b - bloom_score
    elif bloom_score > max_b: return bloom_score - max_b
    return 0.0 

def get_p_value(series_A, series_B):
    df_pair = pd.concat([series_A, series_B], axis=1).dropna()
    a = df_pair.iloc[:, 0].tolist()
    b = df_pair.iloc[:, 1].tolist()
    if len(a) < 2 or a == b: return 1.0
    try:
        _, p_val = st.wilcoxon(a, b, zero_method='wilcox')
        return p_val
    except: return 1.0

def format_p(p):
    if p < 0.001: return f"{p:.4f} (***)"
    if p < 0.01:  return f"{p:.4f} (**)"
    if p < 0.05:  return f"{p:.4f} (*)"
    return f"{p:.4f} (ns)"

def main():
    flat_data = []
    
    steer_cache = {}
    if os.path.exists(CACHE_STEER):
        with open(CACHE_STEER, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                d = json.loads(line)
                steer_cache[d["id"]] = d["content"]

    fact_cache = {}
    if os.path.exists(CACHE_FACT):
        with open(CACHE_FACT, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                d = json.loads(line)
                fact_cache[d["id"]] = d["content"]

    for task_id, steer_content in steer_cache.items():
        parts = task_id.split("_", 2)
        if len(parts) < 3: continue
        seed, topic, pipeline = parts[0], parts[1], parts[2]
        
        fact_content = fact_cache.get(task_id, {})

        for tier in ["foundational", "intermediate", "advanced"]:
            s_data = steer_content.get(f"{tier}_target", {})
            f_data = fact_content.get(tier, {})
            
            bloom = s_data.get("bloom_score", 0)
            tier_acc = 100.0 if s_data.get("predicted_tier") == tier else 0.0
            fact = f_data.get("fact_score", 0.0) * 100
            
            wc = f_data.get("word_count", 0)
            density = (f_data.get("true_claims", 0) / wc * 100) if wc > 0 else 0.0

            flat_data.append({
                "pair_id": f"{seed}_{topic}_{tier}",
                "pair_id_delta": f"{seed}_{topic}",
                "pipeline": pipeline,
                "run_seed": seed,
                "target_tier": tier,
                "bloom": float(bloom),
                "bce": calculate_bce(tier, bloom),
                "fact": fact,
                "tier_acc": tier_acc,
                "density": density
            })

    df = pd.DataFrame(flat_data)
    seed_tier_means = df.groupby(['pipeline', 'run_seed', 'target_tier']).mean(numeric_only=True).reset_index()

    report = "=========================================================\n"
    report += "   CLINICALLY-MAPPED COGNITIVE COMPLEXITY & SIGNIFICANCE \n"
    report += "=========================================================\n\n"

    # 1. BLOOM'S RAW SCORES
    report += "--- 1. RAW BLOOM SCORES (1-6) BY TIER ---\n"
    for tier in ["foundational", "intermediate", "advanced"]:
        report += f">> TARGET TIER: {tier.upper()}\n"
        for p_key, p_name in PIPELINES.items():
            vals = seed_tier_means[(seed_tier_means['pipeline'] == p_key) & (seed_tier_means['target_tier'] == tier)]['bloom'].tolist()
            report += format_line(p_name, vals) + "\n"
        report += "\n"

    # 2. COGNITIVE SEPARATION DELTAS
    report += "--- 2. COGNITIVE SEPARATION DELTAS (The 'Stair-Step' Spectrum) ---\n"
    delta_configs = [
        ("Δ (Intermediate - Foundational)", "intermediate", "foundational"),
        ("Δ (Advanced - Intermediate)", "advanced", "intermediate"),
        ("Δ TOTAL (Advanced - Foundational)", "advanced", "foundational")
    ]
    
    for delta_label, high_tier, low_tier in delta_configs:
        report += f">> {delta_label}\n"
        for p_key, p_name in PIPELINES.items():
            high_df = seed_tier_means[(seed_tier_means['pipeline'] == p_key) & (seed_tier_means['target_tier'] == high_tier)].sort_values('run_seed')
            low_df = seed_tier_means[(seed_tier_means['pipeline'] == p_key) & (seed_tier_means['target_tier'] == low_tier)].sort_values('run_seed')
            
            high_vals = high_df['bloom'].tolist()
            low_vals = low_df['bloom'].tolist()
            deltas = [h - l for h, l in zip(high_vals, low_vals)]
            report += format_line(p_name, deltas) + "\n"
        report += "\n"

    # 3. STATISTICAL SIGNIFICANCE (WILCOXON)
    report += "=========================================================\n"
    report += "   STATISTICAL SIGNIFICANCE REPORT (WILCOXON P-VALUES) \n"
    report += "=========================================================\n"
    report += "* Significance Levels: (*) p < 0.05 | (**) p < 0.01 | (***) p < 0.001 | (ns) Not Significant\n\n"
    
    pivot_bloom = df.pivot(index='pair_id', columns='pipeline', values='bloom')
    pivot_bce = df.pivot(index='pair_id', columns='pipeline', values='bce')
    pivot_fact = df.pivot(index='pair_id', columns='pipeline', values='fact')
    pivot_tier = df.pivot(index='pair_id', columns='pipeline', values='tier_acc')
    
    # Evaluate density strictly on Advanced tier 
    df_adv = df[df['target_tier'] == 'advanced']
    pivot_density_adv = df_adv.pivot(index='pair_id', columns='pipeline', values='density')
    
    df_adv_bloom = df_adv.set_index(['pair_id_delta', 'pipeline'])['bloom']
    df_fnd_bloom = df[df['target_tier'] == 'foundational'].set_index(['pair_id_delta', 'pipeline'])['bloom']
    df_delta = (df_adv_bloom - df_fnd_bloom).unstack(level='pipeline')

    comparisons = [
        ("SFT (Closed) vs Baseline (Closed)", "p3_sft_closed", "p1_baseline_closed"),
        ("SFT (Open) vs Baseline (Open)",     "p4_sft_open",   "p2_baseline_open"),
        ("SFT (Open) vs SFT (Closed)",        "p4_sft_open",   "p3_sft_closed"),
        ("Baseline (Open) vs Baseline (Closed)", "p2_baseline_open", "p1_baseline_closed")
    ]

    for comp_name, m_a, m_b in comparisons:
        report += f"📊 COMPARISON: {comp_name}\n"
        report += "-" * 57 + "\n"
        p_bloom = get_p_value(pivot_bloom[m_a], pivot_bloom[m_b])
        p_bce   = get_p_value(pivot_bce[m_a], pivot_bce[m_b])
        p_fact  = get_p_value(pivot_fact[m_a], pivot_fact[m_b])
        p_tier  = get_p_value(pivot_tier[m_a], pivot_tier[m_b])
        p_dens  = get_p_value(pivot_density_adv[m_a], pivot_density_adv[m_b])
        p_delta = get_p_value(df_delta[m_a], df_delta[m_b])
        
        report += f"  • Raw Cognitive Complexity (Bloom) : p = {format_p(p_bloom)}\n"
        report += f"  • Cognitive Calibration Error (BCE): p = {format_p(p_bce)}\n"
        report += f"  • Cognitive Separation (Δ Bloom)   : p = {format_p(p_delta)}\n"
        report += f"  • Factuality Accuracy              : p = {format_p(p_fact)}\n"
        report += f"  • Factual Density (Advanced Tier)  : p = {format_p(p_dens)}\n"
        report += f"  • Tier Match Rate                  : p = {format_p(p_tier)}\n\n"

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"✅ Significance report saved to {REPORT_FILE}")

if __name__ == "__main__":
    main()
