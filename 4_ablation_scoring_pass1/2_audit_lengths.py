import os
import json
import re
import statistics

RESULTS_FILE = "phase37_4way_multiseed_results.jsonl"
CACHE_BASELINE = "phase37_eval_cache_baseline.jsonl"
CACHE_SFT = "phase37_eval_cache_sft.jsonl"

TIER_BOUNDS = {
    "foundational": {"label": "< 45 words", "check": lambda wc: wc < 45},
    "intermediate": {"label": "60 - 110 words", "check": lambda wc: 60 <= wc <= 110},
    "advanced":     {"label": ">= 150 words", "check": lambda wc: wc >= 150},
}

PIPELINES = [
    ("p1_baseline_closed", "Baseline (Closed-Book)"),
    ("p2_baseline_open",   "Baseline (Open-Book)"),
    ("p3_sft_closed",      "Phase 37 SFT (Closed-Book)"),
    ("p4_sft_open",        "Phase 37 SFT (Open-Book)")
]

def extract_answer(text):
    if not text:
        return ""
    try:
        match = re.search(r'\[ANSWER\](.*)(?:\[/ANSWER\]|$)', text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    except:
        pass
    return text.strip()

def load_data():
    # Structure: stats[pipeline][tier] = [word_counts]
    data_store = {p_key: {tier: [] for tier in TIER_BOUNDS} for p_key, _ in PIPELINES}

    if os.path.exists(RESULTS_FILE) and os.path.getsize(RESULTS_FILE) > 0:
        print(f"📥 Loading data from compiled matrix: {RESULTS_FILE}")
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                row = json.loads(line)
                tiers_data = row.get("pipeline_outputs", {})
                for tier in TIER_BOUNDS:
                    tier_outputs = tiers_data.get(tier, {})
                    for p_key, _ in PIPELINES:
                        raw_text = tier_outputs.get(p_key, "")
                        ans = extract_answer(raw_text)
                        if ans:
                            data_store[p_key][tier].append(len(ans.split()))
    else:
        print("📥 Compiled file not found or empty. Loading directly from cache files...")
        def parse_cache(c_file, p_closed, p_open):
            if not os.path.exists(c_file): return
            with open(c_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    row = json.loads(line)
                    task_id = row.get("id", "")
                    content = row.get("content", {})
                    
                    # task_id format: {seed}_{topic}_{tier}
                    parts = task_id.split("_")
                    tier = parts[-1].lower()
                    if tier in TIER_BOUNDS:
                        ans_c = extract_answer(content.get("closed", ""))
                        ans_o = extract_answer(content.get("open", ""))
                        if ans_c: data_store[p_closed][tier].append(len(ans_c.split()))
                        if ans_o: data_store[p_open][tier].append(len(ans_o.split()))

        parse_cache(CACHE_BASELINE, "p1_baseline_closed", "p2_baseline_open")
        parse_cache(CACHE_SFT, "p3_sft_closed", "p4_sft_open")

    return data_store

def main():
    data = load_data()

    print("\n" + "=" * 90)
    print("📊 4-WAY ABLATION LENGTH & BOUNDARY COMPLIANCE AUDIT")
    print("=" * 90)

    for tier in ["foundational", "intermediate", "advanced"]:
        bounds = TIER_BOUNDS[tier]
        print(f"\n🔹 TARGET TIER: {tier.upper()} (Target: {bounds['label']})")
        print("-" * 90)
        print(f"{'Pipeline':<32} | {'Count':<6} | {'Mean (SD)':<15} | {'Median':<7} | {'Range':<11} | {'Compliance %':<12}")
        print("-" * 90)

        for p_key, p_name in PIPELINES:
            counts = data[p_key][tier]
            if not counts:
                print(f"{p_name:<32} | {'0':<6} | {'N/A':<15} | {'N/A':<7} | {'N/A':<11} | {'N/A':<12}")
                continue

            n = len(counts)
            mean_val = statistics.mean(counts)
            median_val = statistics.median(counts)
            stdev_val = statistics.stdev(counts) if n > 1 else 0.0
            min_val = min(counts)
            max_val = max(counts)
            
            passed = sum(1 for c in counts if bounds["check"](c))
            compliance_pct = (passed / n) * 100

            mean_sd_str = f"{mean_val:.1f} (±{stdev_val:.1f})"
            range_str = f"{min_val} - {max_val}"
            comp_str = f"{compliance_pct:>5.1f}% ({passed}/{n})"

            print(f"{p_name:<32} | {n:<6} | {mean_sd_str:<15} | {median_val:<7.1f} | {range_str:<11} | {comp_str:<12}")

    print("=" * 90)

if __name__ == "__main__":
    main()
