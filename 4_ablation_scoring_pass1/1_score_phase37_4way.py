import os
import requests
import json
import time
import re
import pandas as pd
import numpy as np
import scipy.stats as st
import concurrent.futures
import threading
from tqdm import tqdm
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)

OLLAMA_ENDPOINT = "http://127.0.0.1:11434/api/generate"
FACTUALITY_MODEL = "medgemma:27b"
STEERABILITY_MODEL = "hf.co/rex099/RISE-Judge-Qwen2.5-32B-GGUF:Q4_K_M"

INPUT_FILE = "phase37_4way_multiseed_results.jsonl"
OUTPUT_FILE = "phase37_4way_matrix_scored.jsonl"
REPORT_FILE = "phase37_4way_matrix_report.txt"

CACHE_FACT_DENSITY = "phase37_eval_cache_fact_density_4way.jsonl"
CACHE_STEER = "phase37_eval_cache_steerability_4way.jsonl"
file_lock = threading.Lock()

def clean_json_parse(raw_text):
    if not raw_text: return None
    cleaned = re.sub(r'```(?:json)?\s*', '', raw_text, flags=re.IGNORECASE)
    cleaned = re.sub(r'```\s*$', '', cleaned).strip()
    try: return json.loads(cleaned)
    except: pass
    try:
        match = re.search(r'(\{.*\}|\[.*\])', cleaned, re.DOTALL)        
        if match: return json.loads(match.group(1))
    except: pass
    return None

def call_ollama(prompt, target_model, keep_alive="5m", max_tokens=1024, temp=0.0):
    payload = {
        "model": target_model, "prompt": prompt, "stream": False, "format": "json",
        "keep_alive": keep_alive, "options": {"temperature": temp, "num_predict": max_tokens, "num_ctx": 4096}
    }
    for _ in range(3):
        try:
            res = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=600)
            return res.json().get("response", "").strip()
        except: time.sleep(3)
    return None

def parse_raw_generation(text):
    try:
        q = re.search(r'\[QUESTION\](.*?)\[ANSWER\]', text, re.DOTALL | re.IGNORECASE).group(1).strip()
        a = re.search(r'\[ANSWER\](.*)(?:\[/ANSWER\]|$)', text, re.DOTALL | re.IGNORECASE).group(1).strip()
        return q, a
    except: return "Clinical Topic", text

def check_length_compliance(tier, text):
    try: wc = len(text.split("[ANSWER]")[1].strip().split())
    except: wc = len(text.split())
    if tier == "foundational": return 1.0 if wc < 45 else 0.0
    if tier == "intermediate": return 1.0 if 60 <= wc <= 110 else 0.0
    if tier == "advanced": return 1.0 if wc >= 150 else 0.0
    return 0.0

def load_cache(filepath):
    cache = {}
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    cache[data["id"]] = data["content"]
    return cache

def append_to_cache(filepath, data_dict):
    with file_lock:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(data_dict) + "\n")

def evaluate_factuality_and_density(question, answer):
    ext_prompt = f"""You are an elite clinical pathologist. Break down this QA pair into explicit, standalone atomic claims.
CRITICAL INSTRUCTION: Extract ONLY high-yield medical facts (e.g., genetic mutations, IHC markers, distinct morphological hallmarks, diagnostic criteria). 
DO NOT extract generic conversational text, basic anatomy, or vague filler.
Assign a weight (1.0 for specific molecular/morphological markers, 0.5 for general clinical symptoms).
Output ONLY JSON. Schema: {{ "claims": [ {{"claim": "...", "weight": 1.0}} ] }}

QUESTION: {question}
ANSWER: {answer}"""
    
    raw_ext = call_ollama(ext_prompt, FACTUALITY_MODEL)
    parsed_ext = clean_json_parse(raw_ext)
    claims_data = parsed_ext.get("claims", []) if (parsed_ext and isinstance(parsed_ext, dict)) else []
    valid_claims = [c for c in claims_data if isinstance(c, dict) and c.get("claim")]
    
    if not valid_claims:
        return {"word_count": len(answer.split()), "true_claims": 0, "total_claims": 0, "fact_score": 0.0, "auditable_claims": []}
    
    claims_text = [c.get("claim") for c in valid_claims]
    fact_prompt = f"""Evaluate if each clinical claim is completely medically accurate (true) or contains a medical hallucination/inaccuracy (false) based on WHO consensus.
Output ONLY JSON. Schema: {{ "evaluations": [ {{"is_accurate": true}} ] }}

CLAIMS: {json.dumps(claims_text)}"""
    
    raw_fact = call_ollama(fact_prompt, FACTUALITY_MODEL)
    parsed_fact = clean_json_parse(raw_fact)
    
    evals = []
    if parsed_fact and isinstance(parsed_fact, dict):
        evals = parsed_fact.get("evaluations", [])
    
    auditable_claims = []
    total_weight, earned_weight = 0.0, 0.0
    true_claims = 0

    for i, claim_dict in enumerate(valid_claims):
        weight = float(claim_dict.get("weight", 1.0))
        is_accurate = evals[i].get("is_accurate", False) if i < len(evals) else False
        
        auditable_claims.append({
            "claim": claim_dict.get("claim"),
            "weight": weight,
            "is_accurate": is_accurate
        })
        
        total_weight += weight
        if is_accurate: 
            earned_weight += weight
            true_claims += 1

    fact_score = earned_weight / total_weight if total_weight > 0 else 0.0
    
    return {
        "word_count": len(answer.split()),
        "true_claims": true_claims,
        "total_claims": len(valid_claims),
        "fact_score": fact_score,
        "auditable_claims": auditable_claims
    }

def run_pass1_factuality(task):
    results = {}
    for tier in ["foundational", "intermediate", "advanced"]:
        text = task[f"text_{tier[0]}"]
        q, a = parse_raw_generation(text)
        results[tier] = evaluate_factuality_and_density(q, a)
    append_to_cache(CACHE_FACT_DENSITY, {"id": task["task_id"], "content": results})
    return task["task_id"]

def evaluate_steerability(topic, text_f, text_i, text_a):
    wc_f = len((text_f.split("[ANSWER]")[1].strip() if "[ANSWER]" in text_f else text_f).split())
    wc_i = len((text_i.split("[ANSWER]")[1].strip() if "[ANSWER]" in text_i else text_i).split())
    wc_a = len((text_a.split("[ANSWER]")[1].strip() if "[ANSWER]" in text_a else text_a).split())
    full_prompt = f"""You are an elite Medical Board Examiner.
An AI model was asked to explain the clinical topic "{topic}" at three different difficulty tiers.
You must evaluate the true Pedagogical Tier and Bloom's Cognitive Complexity of each response side-by-side.
CRITICAL INSTRUCTION: Contrast the cognitive depth across the three texts!
- Grade each response purely on its own merit based on the definitions below. Do not assume their intended tier.
- If a text is just a wordier version of another without adding exhaustive multi-system synthesis or deep genetics, it has a LOW Bloom score (1-2) and its true tier is "foundational" despite its length.
CLINICAL BLOOM'S SCALE:
- Level 1-2: Foundational (Definitions, simple processes)
- Level 3-4: Intermediate (Diagnostic rules, IHC panels)
- Level 5-6: Advanced (Clinical judgment, multi-modal synthesis, dense genetics)
[RESPONSE A] (Word Count: {wc_f}):\n{text_f}
[RESPONSE B] (Word Count: {wc_i}):\n{text_i}
[RESPONSE C] (Word Count: {wc_a}):\n{text_a}
Output ONLY JSON:
{{
  "response_A": {{"predicted_tier": "foundational|intermediate|advanced", "bloom_score": <int 1-6>}},
  "response_B": {{"predicted_tier": "foundational|intermediate|advanced", "bloom_score": <int 1-6>}},
  "response_C": {{"predicted_tier": "foundational|intermediate|advanced", "bloom_score": <int 1-6>}}
}}"""
    raw = call_ollama(full_prompt, STEERABILITY_MODEL, max_tokens=300)
    parsed = clean_json_parse(raw)
    results = {
        "foundational_target": {"predicted_tier": "failed", "bloom_score": 0},
        "intermediate_target": {"predicted_tier": "failed", "bloom_score": 0},
        "advanced_target": {"predicted_tier": "failed", "bloom_score": 0}
    }
    if parsed and isinstance(parsed, dict):
        mapping = {"response_a": "foundational_target", "response_b": "intermediate_target", "response_c": "advanced_target"}
        parsed_lower = {k.lower(): v for k, v in parsed.items()}
        for blind_key, target_key in mapping.items():
            if blind_key in parsed_lower:
                raw_tier = parsed_lower[blind_key].get("predicted_tier")
                tier = str(raw_tier).lower() if raw_tier else "failed"
                if tier not in ["foundational", "intermediate", "advanced"]: tier = "failed"
                try: bloom = int(parsed_lower[blind_key].get("bloom_score", 0))
                except: bloom = 0
                results[target_key] = {"predicted_tier": tier, "bloom_score": max(0, min(6, bloom))}
    return results

def run_pass2_steerability(task):
    results = evaluate_steerability(task["topic"], task["text_f"], task["text_i"], task["text_a"])
    append_to_cache(CACHE_STEER, {"id": task["task_id"], "content": results})
    return task["task_id"]

def calc_stats(data_list):
    arr = np.array(data_list)
    if len(arr) < 2: return np.mean(arr), 0.0, 0.0
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    ci_margin = st.t.ppf(0.975, len(arr)-1) * st.sem(arr)
    return mean, std, ci_margin

def main():
    print(f"📥 Loading generation data from {INPUT_FILE}...")
    tasks = []
    pipelines = ["p1_baseline_closed", "p2_baseline_open", "p3_sft_closed", "p4_sft_open"]
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            topic, seed = row["seed_topic"], row["run_seed"]
            tiers = row["pipeline_outputs"]
            for p in pipelines:
                tasks.append({
                    "task_id": f"{seed}_{topic}_{p}", "topic": topic, "seed": seed, "pipeline": p,
                    "text_f": tiers["foundational"][p], "text_i": tiers["intermediate"][p], "text_a": tiers["advanced"][p]
                })

    print(f"\n[PASS 1/2] Running Factuality & Density with {FACTUALITY_MODEL}...")
    fact_cache = load_cache(CACHE_FACT_DENSITY)
    pending_fact = [t for t in tasks if t["task_id"] not in fact_cache]
    if pending_fact:
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futs = [executor.submit(run_pass1_factuality, t) for t in pending_fact]
            for _ in tqdm(concurrent.futures.as_completed(futs), total=len(pending_fact), desc="Factuality/Density"): pass
    
    print(f"\n[PASS 2/2] Running Steerability with {STEERABILITY_MODEL}...")
    steer_cache = load_cache(CACHE_STEER)
    pending_steer = [t for t in tasks if t["task_id"] not in steer_cache]
    if pending_steer:
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futs = [executor.submit(run_pass2_steerability, t) for t in pending_steer]
            for _ in tqdm(concurrent.futures.as_completed(futs), total=len(pending_steer), desc="Steerability"): pass
    
    print("\n📊 Aggregating Results & Generating Report...")
    fact_cache = load_cache(CACHE_FACT_DENSITY)
    steer_cache = load_cache(CACHE_STEER)
    
    flat_data = []
    for t in tasks:
        tid = t["task_id"]
        for tier in ["foundational", "intermediate", "advanced"]:            
            text = t[f"text_{tier[0]}"]
            steer_data = steer_cache.get(tid, {}).get(f"{tier}_target", {"predicted_tier": "failed", "bloom_score": 0})
            fact_data = fact_cache.get(tid, {}).get(tier, {})
            
            wc = fact_data.get("word_count", len(text.split()))
            true_claims = fact_data.get("true_claims", 0)
            density_per_100 = (true_claims / wc * 100) if wc > 0 else 0.0
            
            flat_data.append({
                "pipeline": t["pipeline"], "run_seed": t["seed"], "target_tier": tier,
                "fact": fact_data.get("fact_score", 0.0) * 100,
                "density": density_per_100,
                "true_claims": true_claims,
                "tier_acc": 100.0 if steer_data["predicted_tier"] == tier else 0.0,
                "len_comp": check_length_compliance(tier, text) * 100
            })
            
    df_flat = pd.DataFrame(flat_data)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for row in flat_data:
            out_f.write(json.dumps(row) + "\n")
    overall_seed_means = df_flat.groupby(['pipeline', 'run_seed']).mean(numeric_only=True).reset_index()
    tier_seed_means = df_flat.groupby(['pipeline', 'run_seed', 'target_tier']).mean(numeric_only=True).reset_index()
    
    display_names = {
        "p1_baseline_closed": "Baseline (Closed-Book)",
        "p2_baseline_open": "Baseline (Open-Book Optimal)",
        "p3_sft_closed": "Phase 37 SFT (Closed-Book)",
        "p4_sft_open": "Phase 37 SFT (Open-Book Optimal)"
    }
    
    def format_stat_line(name, data):
        m, sd, ci = calc_stats(data)
        return f"{name:<35} : {m:>6.2f} ± {ci:>5.2f}  (SD: {sd:>5.2f})"
        
    report = "=========================================================\n"
    report += "   PHASE 37 (4-WAY) ABLATION MATRIX REPORT \n"
    report += "=========================================================\n\n"
    
    report += "--- 1. FACTUALITY ACCURACY % (Weight of Accurate Claims) ---\n"
    for p in display_names.keys():
        data = overall_seed_means[overall_seed_means['pipeline'] == p]['fact'].tolist()
        report += format_stat_line(display_names[p], data) + "%\n"

    report += "\n--- 2. CLINICAL FACTUAL DENSITY (Accurate Claims / 100 words) ---\n"
    report += "*Evaluated on the ADVANCED tier.*\n"
    for p in display_names.keys():
        data = tier_seed_means[(tier_seed_means['pipeline'] == p) & (tier_seed_means['target_tier'] == 'advanced')]['density'].tolist()
        report += format_stat_line(display_names[p], data) + "\n"

    report += "\n--- 3. ABSOLUTE CLAIM VOLUME (True Claims per Card) ---\n"
    report += "*Evaluated on the ADVANCED tier.*\n"
    for p in display_names.keys():
        data = tier_seed_means[(tier_seed_means['pipeline'] == p) & (tier_seed_means['target_tier'] == 'advanced')]['true_claims'].tolist()
        report += format_stat_line(display_names[p], data) + "\n"

    report += "\n--- 4. TIER MATCH RATE % (Steerability) ---\n"
    for p in display_names.keys():
        data = overall_seed_means[overall_seed_means['pipeline'] == p]['tier_acc'].tolist()
        report += format_stat_line(display_names[p], data) + "%\n"

    report += "\n--- 5. LENGTH COMPLIANCE % (Strict Word Boundaries) ---\n"
    for p in display_names.keys():
        data = overall_seed_means[overall_seed_means['pipeline'] == p]['len_comp'].tolist()
        report += format_stat_line(display_names[p], data) + "%\n"

    with open(REPORT_FILE, "w", encoding="utf-8") as rep_f: 
        rep_f.write(report)
    print(report)
    print(f"\n✅ Auditable Claims permanently saved to {CACHE_FACT_DENSITY}")

if __name__ == "__main__":
    main()
