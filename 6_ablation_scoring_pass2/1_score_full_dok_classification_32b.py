import os
import requests
import json
import time
import re
import random
import concurrent.futures
import threading
from tqdm import tqdm
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)

# ==========================================
# 1. CONFIGURATION
# ==========================================
OLLAMA_ENDPOINT = "http://127.0.0.1:11434/api/generate"
JUDGE_32B_MODEL = "hf.co/rex099/RISE-Judge-Qwen2.5-32B-GGUF:Q4_K_M"

INPUT_FILE = "phase37_4way_multiseed_results.jsonl"
OUTPUT_FILE = "phase37_final_dok_scored_32b.jsonl"
REPORT_FILE = "phase37_final_dok_report_32b.txt"
CACHE_FILE = "phase37_final_dok_cache_32b.jsonl"

file_lock = threading.Lock()

PIPELINE_MAP = {
    "Baseline_Closed": "p1_baseline_closed",
    "Baseline_Open": "p2_baseline_open",
    "SFT_Closed": "p3_sft_closed",
    "SFT_Open": "p4_sft_open"
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
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

def call_ollama(prompt):
    payload = {
        "model": JUDGE_32B_MODEL, 
        "prompt": prompt, 
        "stream": False, 
        "format": "json",
        "options": {"temperature": 0.0, "num_predict": 200, "num_ctx": 4096}
    }
    for _ in range(3):
        try:
            res = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=300)
            return res.json().get("response", "").strip()
        except: time.sleep(3)
    return None

def load_cache():
    cache = set()
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try: cache.add(json.loads(line)["task_id"])
                    except: pass
    return cache

def append_to_cache(filepath, data_dict):
    with file_lock:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(data_dict) + "\n")

# ==========================================
# 3. EVALUATION LOGIC
# ==========================================
def evaluate_dok_blind_32b(text):
    # UPDATED PROMPT: Incorporating pathology anchors and qualitative audit learnings
    prompt = f"""You are an elite Medical Board Examiner and Pathology Education Auditor.
Classify the cognitive complexity of the following medical flashcard answer into EXACTLY ONE standardized level based on Webb's Depth of Knowledge (DOK) and the SOLO Taxonomy, calibrated specifically for pathology education.

[PATHOLOGY COGNITIVE RUBRIC]
"Level 1" (Foundational / DOK 1 / Unistructural):
- Target: MS1 / Basic Rote Recall.
- Scope: Single-entity identification, basic definitions, classic microscopic/gross morphological hallmarks (e.g., psammoma bodies, spindle cells), or isolated primary stains.
- CRITICAL EXCEPTION: Mentioning a basic comparative marker (e.g., distinguishing BCC vs. SCC) purely to establish a foundational definition DOES NOT elevate the text to Level 2. It remains Level 1.

"Level 2" (Intermediate / DOK 2 / Relational):
- Target: Pathology Resident / Diagnostic Application.
- Scope: COMPLEX differential diagnosis, actively evaluating MULTIPLE distinct entities across MULTIPLE criteria, interpreting multi-marker IHC staining panels (+/- patterns), or clinical grading/staging.

"Level 3" (Advanced / DOK 3-4 / Extended Abstract):
- Target: Hematopathology/Molecular Fellow / Attending Oncologist.
- Scope: Multi-variable molecular synthesis, detailed genomic alterations/fusions, downstream signaling cascades, and specific targeted oncology therapeutics (e.g., TKIs, targeted inhibitors).

[FLASHCARD ANSWER]
{text}

Output ONLY valid JSON:
{{
  "predicted_level": "Level 1" | "Level 2" | "Level 3",
  "reasoning": "Brief 1-sentence justification identifying which category criteria were met."
}}"""
    
    raw = call_ollama(prompt)
    parsed = clean_json_parse(raw)
    result = {"predicted_level": "Unknown", "reasoning": "Parse fail."}
    
    if parsed and isinstance(parsed, dict):
        result["reasoning"] = parsed.get("reasoning", "")
        pred_raw = str(parsed.get("predicted_level", "Unknown")).strip()
        
        if "1" in pred_raw: result["predicted_level"] = "Level 1"
        elif "2" in pred_raw: result["predicted_level"] = "Level 2"
        elif "3" in pred_raw: result["predicted_level"] = "Level 3"
        
    return result

def process_task(task):
    task_id = task["task_id"]
    text = task["text"]
    
    eval_res = evaluate_dok_blind_32b(text)
    
    result = {
        "task_id": task_id,
        "topic": task["topic"],
        "seed": task["seed"],
        "pipeline": task["pipeline"],
        "target_tier": task["target_tier"],
        "eval": eval_res
    }
    
    with file_lock:
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result) + "\n")
            
    append_to_cache(CACHE_FILE, {"task_id": task_id})
    return True

# ==========================================
# 4. ORCHESTRATION
# ==========================================
def main():
    print(f"📥 Loading full dataset from {INPUT_FILE}...")
    
    tasks = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            
            for tier in ["foundational", "intermediate", "advanced"]:
                tier_data = row.get("pipeline_outputs", {}).get(tier, {})
                for display_name, internal_key in PIPELINE_MAP.items():
                    text = tier_data.get(internal_key, "")
                    if text:
                        tasks.append({
                            "task_id": f"{row['run_seed']}_{row['seed_topic']}_{tier}_{display_name}",
                            "topic": row["seed_topic"],
                            "seed": row["run_seed"],
                            "pipeline": display_name,
                            "target_tier": tier,
                            "text": text
                        })
            
    print(f"🚀 Prepared {len(tasks)} tasks for final DOK classification using {JUDGE_32B_MODEL}.")
    cached_tasks = load_cache()
    pending_tasks = [t for t in tasks if t['task_id'] not in cached_tasks]
    
    if len(pending_tasks) < len(tasks):
        print(f"🔄 Resuming from cache... {len(pending_tasks)} remaining to score.")
    else:
        open(OUTPUT_FILE, 'w').close()
        open(CACHE_FILE, 'w').close()
        
    if pending_tasks:
        random.shuffle(pending_tasks)
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futs = [executor.submit(process_task, t) for t in pending_tasks]
            for fut in tqdm(concurrent.futures.as_completed(futs), total=len(pending_tasks), desc="Final 32B DOK Audit"):
                fut.result()
                
    print("\n📊 Aggregating Final Dataset Confusion Matrix...")
    
    matrix = {p: {"foundational": {"Level 1": 0, "Level 2": 0, "Level 3": 0, "Unknown": 0},
                  "intermediate": {"Level 1": 0, "Level 2": 0, "Level 3": 0, "Unknown": 0},
                  "advanced":     {"Level 1": 0, "Level 2": 0, "Level 3": 0, "Unknown": 0}}
              for p in PIPELINE_MAP.keys()}
    
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            p = d["pipeline"]
            t = d["target_tier"]
            pred = d["eval"].get("predicted_level", "Unknown")
            if p in matrix and t in matrix[p] and pred in matrix[p][t]:
                matrix[p][t][pred] += 1

    report = "=========================================================\n"
    report += f"   PHASE 37: FINAL CALIBRATED DOK CLASSIFICATION (N={len(tasks)})\n"
    report += "=========================================================\n"
    report += "* Target Mapping: Foundational = Level 1 | Intermediate = Level 2 | Advanced = Level 3\n\n"
    
    for p in PIPELINE_MAP.keys():
        report += f"--- {p} Confusion Matrix ---\n"
        report += f"                     | Pred Lvl 1 | Pred Lvl 2 | Pred Lvl 3 |\n"
        report += f"-----------------------------------------------------------\n"
        
        for t, expected in zip(["foundational", "intermediate", "advanced"], ["Level 1", "Level 2", "Level 3"]):
            counts = matrix[p][t]
            t_label = t.capitalize()[:12]
            
            total = counts["Level 1"] + counts["Level 2"] + counts["Level 3"]
            matches = counts[expected]
            match_rate = (matches / total * 100) if total > 0 else 0
            
            report += f" Target: {t_label:<11} | {counts['Level 1']:>10} | {counts['Level 2']:>10} | {counts['Level 3']:>10} | -> Match: {match_rate:.1f}%\n"
        report += "\n"

    with open(REPORT_FILE, "w", encoding="utf-8") as rep_f: rep_f.write(report)
    print(report)

if __name__ == "__main__":
    main()
