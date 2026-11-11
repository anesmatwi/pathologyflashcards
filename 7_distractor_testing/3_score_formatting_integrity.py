import os
import json
import pandas as pd
import re

# === CONFIGURATION ===
INPUT_FILE = "phase37_4way_multiseed_results.jsonl"
REPORT_FILE = "phase37_formatting_integrity_report.txt"

PIPELINES = {
    "p1_baseline_closed": "Baseline (Closed-Book)",
    "p2_baseline_open": "Baseline (Open-Book Optimal)",
    "p3_sft_closed": "Phase 37 SFT (Closed-Book)",
    "p4_sft_open": "Phase 37 SFT (Open-Book Optimal)"
}

# === STRUCTURAL INTEGRITY EVALUATION ===
def evaluate_structural_integrity(text):
    if not text:
        return 0.0, True, True

    sci_score = 1.0
    filler_v = False
    tag_v = False
    
    fillers = [
        "here is your", "certainly", "i hope this helps", "sure!", 
        "as requested", "here's a", "happy to help", "here is a", 
        "flashcard:", "front:", "back:"
    ]
    
    # Use strict regex boundaries to prevent "there is a" from matching "here is a"
    text_lower = text.lower()
    for f in fillers:
        if re.search(r'(?<!\w)' + re.escape(f) + r'(?!\w)', text_lower):
            filler_v = True
            sci_score -= 0.5
            break
            
    # Check strict syntax boundaries
    q_idx = text.find("[QUESTION]")
    a_idx = text.find("[ANSWER]")
    
    if q_idx == -1 or a_idx == -1 or q_idx >= a_idx:
        tag_v = True
        sci_score -= 0.5
        
    return max(0.0, sci_score), filler_v, tag_v

# === MAIN EXECUTION ===
def main():
    print(f"📥 Loading generated dataset from {INPUT_FILE}...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found.")
        return

    flat_data = []
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            
            for tier in ["foundational", "intermediate", "advanced"]:
                tier_outputs = data.get("pipeline_outputs", {}).get(tier, {})
                
                for p_key, p_name in PIPELINES.items():
                    text = tier_outputs.get(p_key, "")
                    sci_score, filler_v, tag_v = evaluate_structural_integrity(text)
                    
                    flat_data.append({
                        "seed_topic": data.get("seed_topic", "unknown"),
                        "target_tier": tier,
                        "pipeline": p_name,
                        "sci_score": sci_score * 100,
                        "filler_violation": (1 if filler_v else 0) * 100,
                        "tag_violation": (1 if tag_v else 0) * 100
                    })
                    
    df = pd.DataFrame(flat_data)
    
    if df.empty:
        print("⚠️ No valid output data found.")
        return
        
    print("📊 Aggregating Structural Integrity Metrics...")
    
    report = "=========================================================\n"
    report += "   PHASE 37: FORMATTING & STRUCTURAL INTEGRITY AUDIT \n"
    report += "=========================================================\n\n"
    
    report += "--- 1. OVERALL STRUCTURAL INTEGRITY (%) ---\n"
    report += "* Score based on strict adherence to [QUESTION]/[ANSWER] syntax and zero conversational filler.\n\n"
    
    for p_name in PIPELINES.values():
        mean_sci = df[df['pipeline'] == p_name]['sci_score'].mean()
        report += f"{p_name:<35} : {mean_sci:>5.1f}%\n"
        
    report += "\n--- 2. GRANULAR VIOLATION BREAKDOWN (Lower is Better) ---\n"
    
    for p_name in PIPELINES.values():
        sub_df = df[df['pipeline'] == p_name]
        mean_filler = sub_df['filler_violation'].mean()
        mean_tag = sub_df['tag_violation'].mean()
        
        report += f"[{p_name}]\n"
        report += f"  - Conversational Leaks (Filler)  : {mean_filler:>5.1f}%\n"
        report += f"  - Syntax Tag Failure (Missing)   : {mean_tag:>5.1f}%\n\n"
        
    report += "--- 3. INTEGRITY BY COGNITIVE TIER ---\n"
    for tier in ["foundational", "intermediate", "advanced"]:
        report += f">> TARGET TIER: {tier.upper()}\n"
        for p_name in PIPELINES.values():
            mean_sci = df[(df['pipeline'] == p_name) & (df['target_tier'] == tier)]['sci_score'].mean()
            report += f"  {p_name:<33} : {mean_sci:>5.1f}%\n"
        report += "\n"

    report += "=========================================================\n"

    with open(REPORT_FILE, "w", encoding="utf-8") as rep_f:
        rep_f.write(report)
        
    print(report)
    print(f"✅ Full Formatting & Integrity report saved to {REPORT_FILE}")

if __name__ == "__main__":
    main()
