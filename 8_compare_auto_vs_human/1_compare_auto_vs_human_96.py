import os
import re
import json
import numpy as np
import pandas as pd
import scipy.stats as st

# === INPUT FILES ===
MULTISEED_FILE = "phase37_4way_multiseed_results.jsonl"
BLINDED_CARDS_FILE = "blinded_eval_set.jsonl"
HUMAN_EVAL_FILE = "dr_honeine_evaluations.jsonl"

CACHE_FACT = "phase37_eval_cache_fact_density_4way.jsonl"
CACHE_STEER = "phase37_eval_cache_steerability_4way.jsonl"
DOK_SCORED_FILE = "phase37_final_dok_scored_32b.jsonl"

REPORT_FILE = "phase37_auto_vs_human_96_report.txt"

PIPELINES = {
    "p1_baseline_closed": "Baseline (Closed-Book)",
    "p2_baseline_open":   "Baseline (Open-Book Optimal)",
    "p3_sft_closed":      "Phase 37 SFT (Closed-Book)",
    "p4_sft_open":        "Phase 37 SFT (Open-Book Optimal)"
}

DOK_PIPE_MAP = {
    "p1_baseline_closed": "Baseline_Closed",
    "p2_baseline_open":   "Baseline_Open",
    "p3_sft_closed":      "SFT_Closed",
    "p4_sft_open":        "SFT_Open"
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

TIER_TO_INT = {"foundational": 1, "intermediate": 2, "advanced": 3}
DOK_TO_INT = {"Level 1": 1, "Level 2": 2, "Level 3": 3}

def extract_qa_400(text):
    """Exact replica of extract_qa from 400_prepare_blinded_sample.py"""
    if not text: return None, None
    q_match = re.search(r'\[QUESTION\]\s*(.*?)\s*\[ANSWER\]', text, re.DOTALL | re.IGNORECASE)
    a_match = re.search(r'\[ANSWER\]\s*(.*?)(?:\[/ANSWER\]|$)', text, re.DOTALL | re.IGNORECASE)
    if q_match and a_match:
        return q_match.group(1).strip(), a_match.group(1).strip()
    try:
        q_part = re.search(r'(?:Question|Clinical Question)[\*:#\s]*(.*?)(?=(?:\n\s*)*[\*#]*\s*Answer[\*:#\s]*)', text, re.DOTALL | re.IGNORECASE)
        a_part = re.search(r'Answer[\*:#\s]*(.*)', text, re.DOTALL | re.IGNORECASE)
        if q_part and a_part:
            q = re.sub(r'[\*#]+$', '', q_part.group(1).strip()).strip()
            return q, a_part.group(1).strip()
    except Exception:
        pass
    return None, None

def parse_leading_int(val):
    if pd.isna(val) or not str(val).strip(): return np.nan
    m = re.match(r"^(\d+)", str(val).strip())
    return float(m.group(1)) if m else np.nan

def main():
    # 1. Reconstruct exact (tier, topic) -> run_seed mapping used by 400_prepare_blinded_sample.py
    topic_tier_to_seed = {}
    with open(MULTISEED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            topic = data.get("seed_topic", "Unknown")
            seed = data.get("run_seed")
            for tier in ["foundational", "intermediate", "advanced"]:
                outputs = data.get("pipeline_outputs", {}).get(tier, {})
                valid_count = 0
                for p in PIPELINES:
                    q, a = extract_qa_400(outputs.get(p, ""))
                    if q and a: valid_count += 1
                if valid_count == 4:
                    # 400_prepare_blinded_sample.py overwrote valid_topics_by_tier[tier][topic] in file order
                    topic_tier_to_seed[(tier, topic)] = seed

    # 2. Load Automated Caches
    fact_cache = {}
    if os.path.exists(CACHE_FACT):
        with open(CACHE_FACT, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    fact_cache[d["id"]] = d["content"]

    steer_cache = {}
    if os.path.exists(CACHE_STEER):
        with open(CACHE_STEER, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    steer_cache[d["id"]] = d["content"]

    dok_cache = {}
    if os.path.exists(DOK_SCORED_FILE):
        with open(DOK_SCORED_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    dok_cache[d["task_id"]] = d["eval"]

    # 3. Load Blinded Cards (for word counts) and Human Evaluations
    cards_by_id = {}
    if os.path.exists(BLINDED_CARDS_FILE):
        with open(BLINDED_CARDS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    c = json.loads(line)
                    cards_by_id[c["eval_id"]] = c

    rows = []
    with open(HUMAN_EVAL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            h = json.loads(line)
            eval_id = h["eval_id"]
            topic = h["topic"]
            tier = h["tier_true"]
            p_key = h["model_true"]
            seed = topic_tier_to_seed.get((tier, topic))

            # Lookup keys in automated caches
            task_id_4way = f"{seed}_{topic}_{p_key}"
            task_id_dok = f"{seed}_{topic}_{tier}_{DOK_PIPE_MAP[p_key]}"

            f_data = fact_cache.get(task_id_4way, {}).get(tier, {})
            s_data = steer_cache.get(task_id_4way, {}).get(f"{tier}_target", {})
            d_data = dok_cache.get(task_id_dok, {})

            true_lvl = TIER_TO_INT[tier]
            
            # Automated predictions
            auto_pass1_tier = TIER_TO_INT.get(s_data.get("predicted_tier", ""), np.nan)
            auto_pass1_bloom = float(s_data.get("bloom_score", np.nan))
            auto_pass2_dok = DOK_TO_INT.get(d_data.get("predicted_level", ""), np.nan)
            auto_fact_pct = float(f_data.get("fact_score", np.nan)) * 100.0
            auto_wc = f_data.get("word_count", len(cards_by_id.get(eval_id, {}).get("a", "").split()))

            # Human ratings
            hum_aud_lvl = AUDIENCE_MAP.get(h.get("tier_assigned"), np.nan)
            hum_bloom_lvl = BLOOM_MAP.get(h.get("bloom_tier"), np.nan)
            hum_fact = parse_leading_int(h.get("factual"))
            hum_util = parse_leading_int(h.get("utility"))
            hum_scope = parse_leading_int(h.get("scope"))

            rows.append({
                "eval_id": eval_id,
                "seed": seed,
                "topic": topic,
                "tier_true": tier,
                "true_lvl": true_lvl,
                "pipeline": p_key,
                "auto_wc": auto_wc,
                "auto_fact_pct": auto_fact_pct,
                "auto_pass1_tier": auto_pass1_tier,
                "auto_pass1_bloom": auto_pass1_bloom,
                "auto_pass1_match": (auto_pass1_tier == true_lvl),
                "auto_pass2_dok": auto_pass2_dok,
                "auto_pass2_match": (auto_pass2_dok == true_lvl),
                "auto_pass2_reason": d_data.get("reasoning", ""),
                "hum_fact": hum_fact,
                "hum_util": hum_util,
                "hum_scope": hum_scope,
                "hum_aud_raw": h.get("tier_assigned", ""),
                "hum_bloom_raw": h.get("bloom_tier", ""),
                "hum_aud_lvl": hum_aud_lvl,
                "hum_bloom_lvl": hum_bloom_lvl,
                "hum_aud_match": (hum_aud_lvl == true_lvl) if pd.notna(hum_aud_lvl) else False,
                "hum_bloom_match": (hum_bloom_lvl == true_lvl) if pd.notna(hum_bloom_lvl) else False,
                "hum_dual_match": (hum_aud_lvl == true_lvl and hum_bloom_lvl == true_lvl) if (pd.notna(hum_aud_lvl) and pd.notna(hum_bloom_lvl)) else False,
                "hum_either_match": (hum_aud_lvl == true_lvl or hum_bloom_lvl == true_lvl) if (pd.notna(hum_aud_lvl) and pd.notna(hum_bloom_lvl)) else False,
                "classifiable": (pd.notna(hum_aud_lvl) and pd.notna(hum_bloom_lvl)),
                "comments": h.get("comments", "")
            })

    df = pd.DataFrame(rows)

    out = []
    out.append("=========================================================================================================")
    out.append("   HEAD-TO-HEAD COMPARISON: AUTOMATED JUDGES vs. PATHOLOGIST ON THE EXACT SAME 96 CARDS (24 TOPICS)")
    out.append("=========================================================================================================\n")

    # SECTION 1: Pipeline-Level Summary on the 96 Cards
    out.append("--- 1. PIPELINE COMPARISON ON THE IDENTICAL 96-CARD SAMPLE (n=24 per pipeline) ---")
    hdr = (f"{'Pipeline':<32} | {'Auto Pass1 Bloom':<16} | {'Auto Pass2 DOK':<15} | "
           f"{'Hum Bloom (Cls/All)':<20} | {'Hum Aud (Cls/All)':<18} | {'Hum Dual (Cls/All)':<18}")
    out.append(hdr)
    out.append("-" * len(hdr))

    for p_key, p_name in PIPELINES.items():
        sub = df[df["pipeline"] == p_key]
        cls_sub = sub[sub["classifiable"]]
        n_all = len(sub)
        n_cls = len(cls_sub)

        ap1 = int(sub["auto_pass1_match"].sum())
        ap2 = int(sub["auto_pass2_match"].sum())

        hb_cls = int(cls_sub["hum_bloom_match"].sum())
        hb_all = int((sub["hum_bloom_lvl"] == sub["true_lvl"]).sum())

        ha_cls = int(cls_sub["hum_aud_match"].sum())
        ha_all = int((sub["hum_aud_lvl"] == sub["true_lvl"]).sum())

        hd_cls = int(cls_sub["hum_dual_match"].sum())
        hd_all = int(((sub["hum_aud_lvl"] == sub["true_lvl"]) & (sub["hum_bloom_lvl"] == sub["true_lvl"])).sum())

        out.append(
            f"{p_name:<32} | {ap1/n_all*100:>5.1f}% ({ap1:>2}/{n_all:<2})   | {ap2/n_all*100:>5.1f}% ({ap2:>2}/{n_all:<2})  | "
            f"{hb_cls/n_cls*100:>5.1f}% [{hb_all/n_all*100:>4.1f}%] ({hb_cls:>2}/{n_cls:<2}) | "
            f"{ha_cls/n_cls*100:>5.1f}% [{ha_all/n_all*100:>4.1f}%]   | "
            f"{hd_cls/n_cls*100:>5.1f}% [{hd_all/n_all*100:>4.1f}%]"
        )

    # SECTION 2: Tier-by-Tier Breakdown on the 96 Cards
    out.append("\n--- 2. TIER-BY-TIER BREAKDOWN ON THE 96 CARDS (n=8 per pipeline per tier) ---")
    for tier in ["foundational", "intermediate", "advanced"]:
        out.append(f"\n>> TARGET TIER: {tier.upper()} (n=8 per pipeline)")
        out.append("-" * len(hdr))
        for p_key, p_name in PIPELINES.items():
            sub = df[(df["pipeline"] == p_key) & (df["tier_true"] == tier)]
            cls_sub = sub[sub["classifiable"]]
            n_all = len(sub)
            n_cls = len(cls_sub)

            ap1 = int(sub["auto_pass1_match"].sum())
            ap2 = int(sub["auto_pass2_match"].sum())
            hb_cls = int(cls_sub["hum_bloom_match"].sum())
            ha_cls = int(cls_sub["hum_aud_match"].sum())
            hd_cls = int(cls_sub["hum_dual_match"].sum())

            hb_str = f"{hb_cls/n_cls*100:>5.1f}% ({hb_cls}/{n_cls})" if n_cls else "N/A"
            ha_str = f"{ha_cls/n_cls*100:>5.1f}% ({ha_cls}/{n_cls})" if n_cls else "N/A"
            hd_str = f"{hd_cls/n_cls*100:>5.1f}% ({hd_cls}/{n_cls})" if n_cls else "N/A"

            out.append(
                f"{p_name:<32} | {ap1/n_all*100:>5.1f}% ({ap1}/{n_all})     | {ap2/n_all*100:>5.1f}% ({ap2}/{n_all})    | "
                f"{hb_str:<20} | {ha_str:<18} | {hd_str:<18}"
            )

    # SECTION 3: Diagnostic Attribution of Discrepancies (Why Auto Matched Cards Failed Human Strict Match)
    out.append("\n--- 3. WHY CARDS THAT PASSED AUTOMATED PASS 2 (DOK/SOLO) DISAGREED WITH HUMAN STRICT MATCH ---")
    for p_key in ["p4_sft_open", "p3_sft_closed", "p2_baseline_open", "p1_baseline_closed"]:
        sub = df[df["pipeline"] == p_key]
        auto_passed = sub[sub["auto_pass2_match"]]
        disagreed = auto_passed[~auto_passed["hum_dual_match"]]

        c_unclass = int((~disagreed["classifiable"]).sum())
        c_bloom_ok_aud_low = int((disagreed["classifiable"] & disagreed["hum_bloom_match"] & (~disagreed["hum_aud_match"])).sum())
        c_aud_ok_bloom_low = int((disagreed["classifiable"] & (~disagreed["hum_bloom_match"]) & disagreed["hum_aud_match"]).sum())
        c_both_low = int((disagreed["classifiable"] & (~disagreed["hum_bloom_match"]) & (~disagreed["hum_aud_match"])).sum())
        c_scope_perfect = int((disagreed["hum_scope"] == 3).sum())

        out.append(f"\n[{PIPELINES[p_key]}]")
        out.append(f"  • Automated Pass 2 (DOK) Matched Target Tier : {len(auto_passed)}/24 ({len(auto_passed)/24*100:.1f}%)")
        out.append(f"  • Also Matched Human Strict Dual-Axis        : {len(auto_passed) - len(disagreed)}/{len(auto_passed)}")
        out.append(f"  • Disagreed on Human Strict Dual-Axis        : {len(disagreed)}/{len(auto_passed)} cards — Breakdown of those {len(disagreed)} cards:")
        out.append(f"      1. Human agreed on Bloom's Task, but rated Target Audience lower (e.g. MS1): {c_bloom_ok_aud_low}")
        out.append(f"      2. Human agreed on Target Audience, but rated Bloom's Task lower           : {c_aud_ok_bloom_low}")
        out.append(f"      3. Human marked Bloom's Task 'Unclassifiable' (ultra-brief list/repetition): {c_unclass}")
        out.append(f"      4. Human rated both Audience and Bloom's different from target tier        : {c_both_low}")
        out.append(f"      * Note: {c_scope_perfect}/{len(disagreed)} ({c_scope_perfect/len(disagreed)*100 if len(disagreed) else 0:.1f}%) of these discordant cards were still rated '3/3 Perfectly Scoped' by the pathologist.")

    # SECTION 4: Card-by-Card Table for SFT Open-Book (n=24) and SFT Closed-Book (n=24)
    out.append("\n--- 4. CARD-BY-CARD HEAD-TO-HEAD LEDGER FOR SFT OPEN-BOOK (p4_sft_open, n=24) ---")
    sft_open = df[df["pipeline"] == "p4_sft_open"].sort_values(["true_lvl", "eval_id"])
    for _, r in sft_open.iterrows():
        out.append(
            f"  Card {int(r['eval_id']):>2} | Tier: {r['tier_true']:<12} ({int(r['auto_wc']):>3}w) | Topic: {r['topic']:<28} | "
            f"AutoP1: L{int(r['auto_pass1_tier'])}(B={int(r['auto_pass1_bloom'])}) | AutoP2: DOK-{int(r['auto_pass2_dok'])} | "
            f"Human Aud: {str(r['hum_aud_lvl']):<3} | Human Bloom: {str(r['hum_bloom_lvl']):<3} | Scope: {int(r['hum_scope'])}/3 | Fact: {int(r['hum_fact'])}/5"
        )

    report_text = "\n".join(out)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(report_text)
    print(f"\n✅ Saved head-to-head 96-card comparison to {REPORT_FILE}")

if __name__ == "__main__":
    main()
