import os
import json
import requests
import re
import time
import threading
import concurrent.futures
from tqdm import tqdm
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)

# ==========================================
# 1. CONFIGURATION
# ==========================================
OLLAMA_ENDPOINT = "http://127.0.0.1:11434/api/generate"
OLLAMA_CHAT_ENDPOINT = "http://127.0.0.1:11434/api/chat"
COMMAND_R_MODEL = "command-r:35b"
MEDGEMMA_MODEL = "medgemma:27b"

CACHE_FILE = "phase37_distractor_rag_cache.jsonl"
REPORT_FILE = "phase37_distractor_rag_report.txt"

file_lock = threading.Lock()

MODELS_TO_TEST = {
    "Baseline_Qwen": "qwen2.5:14b",
    "Phase37_SFT": "pathology_tutor_phase37_sft:latest"
}

DISTRACTOR_DATASET = [
    {
        "target_topic": "Gastrointestinal Stromal Tumor (GIST)",
        "distractor_topic": "Melanoma",
        "distractor_key_features": "BRAF V600E, S100, Melan-A, HMB-45, SOX10, atypical melanocytes, pagetoid spread"
    },
    {
        "target_topic": "Yolk Sac Tumor",
        "distractor_topic": "Meningioma",
        "distractor_key_features": "EMA, PR, psammoma bodies, whorled pattern, dural tail, NF2 mutation"
    },
    {
        "target_topic": "Medullary Thyroid Carcinoma",
        "distractor_topic": "Ewing Sarcoma",
        "distractor_key_features": "t(11;22), EWSR1-FLI1, CD99, small round blue cells, PAS positive glycogen"
    },
    {
        "target_topic": "Chronic Myeloid Leukemia",
        "distractor_topic": "Granulosa Cell Tumor",
        "distractor_key_features": "FOXL2 mutation, Call-Exner bodies, inhibin, calretinin, coffee-bean nuclei"
    },
    {
        "target_topic": "Mantle Cell Lymphoma",
        "distractor_topic": "Clear Cell Renal Cell Carcinoma",
        "distractor_key_features": "VHL gene, 3p deletion, CAIX, PAX8, chicken-wire vasculature, clear cytoplasm"
    },
    {
        "target_topic": "Papillary Thyroid Carcinoma",
        "distractor_topic": "Classic Hodgkin Lymphoma",
        "distractor_key_features": "Reed-Sternberg cells, CD30, CD15, Pax-5 weak, nodular sclerosis, lacunar cells"
    },
    {
        "target_topic": "Small Cell Lung Carcinoma",
        "distractor_topic": "Breast Fibroadenoma",
        "distractor_key_features": "MED12 mutation, biphasic stromal-epithelial proliferation, intracanalicular, pericanalicular"
    },
    {
        "target_topic": "Prostate Adenocarcinoma",
        "distractor_topic": "Glioblastoma",
        "distractor_key_features": "IDH wildtype, EGFR amplification, TERT promoter mutation, pseudopalisading necrosis, microvascular proliferation"
    },
    {
        "target_topic": "Follicular Lymphoma",
        "distractor_topic": "Hepatocellular Carcinoma",
        "distractor_key_features": "Glypican-3, Arginase-1, HepPar-1, bile production, Mallory-Denk bodies, cirrhosis"
    },
    {
        "target_topic": "Seminoma",
        "distractor_topic": "Squamous Cell Carcinoma of Lung",
        "distractor_key_features": "p40, p63, CK5/6, keratin pearls, intercellular bridges"
    }
]

TIERS = ["foundational", "intermediate", "advanced"]

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def get_production_system_prompt(tier):
    if tier == "foundational":
        return "You are a Foundational Pathology Tutor for Year-1 medical students. Your sole purpose is to test basic rote recall, simple definitions, and morphological features. You are FORBIDDEN from discussing multi-disease comparisons, clinical treatments, or advanced molecular pathways."
    elif tier == "intermediate":
        return "You are a Clinical Pathology Instructor focusing on diagnostic application for residents. Your goal is to test the user's ability to contrast distinct pathological entities, interpret diagnostic criteria, and evaluate multi-marker IHC panels."
    elif tier == "advanced":
        return "You are an Elite Molecular Pathologist and Oncologist. Your mandate is exhaustive clinical synthesis. You must evaluate complex genetic markers, dictate targeted oncology treatments, and predict survival outcomes."

def call_ollama_generate(model, prompt, temperature=0.0, fmt=None):
    payload = {
        "model": model, "prompt": prompt, "stream": False, "keep_alive": -1,
        "options": {"temperature": temperature, "num_ctx": 4096}
    }
    if fmt: payload["format"] = fmt
    for attempt in range(3):
        try:
            res = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=240).json()
            return res.get("response", "").strip()
        except: time.sleep(2)
    return None

def call_ollama_chat(system_prompt, user_prompt, model_name, max_tokens=800):
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "keep_alive": -1,
        "options": {"temperature": 0.0, "top_p": 0.9, "num_predict": max_tokens, "num_ctx": 4096}
    }
    for attempt in range(3):
        try:
            res = requests.post(OLLAMA_CHAT_ENDPOINT, json=payload, timeout=300).json()
            return res.get("message", {}).get("content", "").strip()
        except: time.sleep(2)
    return ""

def unload_model(model_name):
    try: requests.post(OLLAMA_ENDPOINT, json={"model": model_name, "keep_alive": 0}, timeout=10)
    except: pass

def load_cache():
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    cache[data["target_topic"]] = data
    for item in DISTRACTOR_DATASET:
        if item["target_topic"] not in cache:
            cache[item["target_topic"]] = {
                "target_topic": item["target_topic"],
                "distractor_topic": item["distractor_topic"],
                "distractor_key_features": item["distractor_key_features"],
                "distractor_context": "",
                "results": {}
            }
    return cache

def save_cache_safely(cache_data):
    with file_lock:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            for data in cache_data.values():
                f.write(json.dumps(data) + "\n")

# ==========================================
# 3. WORKERS
# ==========================================
def worker_generate_distractor_context(topic_data):
    if topic_data.get("distractor_context"): return topic_data
    prompt = f"""You are a medical textbook author. Write an exhaustive, highly academic 250-word textbook excerpt about {topic_data["distractor_topic"]}. 
Include standard clinical pathology details, diagnostic markers, and molecular genetics (e.g., {topic_data['distractor_key_features']}).
Output only the clinical textbook paragraph."""
    context = call_ollama_generate(COMMAND_R_MODEL, prompt, temperature=0.1)
    if context: topic_data["distractor_context"] = context
    return topic_data

def worker_generate_flashcard(task_args):
    target_topic, distractor_context, tier, model_key, actual_model_tag = task_args
    sys_prompt = get_production_system_prompt(tier)
    user_prompt = f"Textbook Context Base:\n{distractor_context}\n\nTopic Concepts: {target_topic}\nWrite the clinical question and answer now."
    answer = call_ollama_chat(sys_prompt, user_prompt, actual_model_tag)
    return target_topic, tier, model_key, answer

def worker_score_adoption(task_args):
    target_topic, distractor_topic, key_features, tier, answer, model_key = task_args
    eval_prompt = f"""You are an elite Medical Auditor evaluating an AI-generated medical flashcard for: "{target_topic}".
The AI was accidentally provided an irrelevant distractor context about "{distractor_topic}" (which involves markers/features like: {key_features}).

Evaluate the AI's generated flashcard answer below:
[FLASHCARD ANSWER]
{answer}

Did the AI mistakenly adopt, blend, or mention any pathology, genetics, IHC stains, or defining features of "{distractor_topic}" into its answer for "{target_topic}"?

Output ONLY JSON. Schema: {{ "adopted_distractor": true/false, "reasoning": "Brief explanation." }}"""
    
    eval_raw = call_ollama_generate(MEDGEMMA_MODEL, eval_prompt, temperature=0.0, fmt="json")
    if not eval_raw: return target_topic, tier, model_key, None
    try:
        parsed_eval = json.loads(re.sub(r'```(?:json)?\s*', '', eval_raw).replace('```', '').strip())
        adopted = 1 if parsed_eval.get("adopted_distractor") else 0
        return target_topic, tier, model_key, adopted
    except: return target_topic, tier, model_key, None

# ==========================================
# 4. ORCHESTRATION
# ==========================================
def main():
    print("🧪 Starting Phase 37 Distractor RAG (Noise Robustness) Test...")
    cache = load_cache()
    
    print(f"\n🧠 STAGE 1: Generating Irrelevant Distractor Contexts [{COMMAND_R_MODEL}]...")
    context_tasks = [data for data in cache.values() if not data.get("distractor_context")]
    if context_tasks:
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            list(tqdm(executor.map(worker_generate_distractor_context, context_tasks), total=len(context_tasks)))
        save_cache_safely(cache)
    unload_model(COMMAND_R_MODEL)

    print(f"\n⚡ STAGE 2: Generating Flashcards under Distractor Noise...")
    for model_key, actual_tag in MODELS_TO_TEST.items():
        gen_tasks = []
        for topic, data in cache.items():
            if not data.get("distractor_context"): continue
            if model_key not in data["results"]: data["results"][model_key] = {}
            
            for tier in TIERS:
                existing_result = data["results"][model_key].get(tier, {})
                if not existing_result or not existing_result.get("answer"):
                    gen_tasks.append((topic, data["distractor_context"], tier, model_key, actual_tag))
        
        if gen_tasks:
            print(f"   -> Loading {actual_tag} for {len(gen_tasks)} tasks...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                for future in tqdm(concurrent.futures.as_completed([executor.submit(worker_generate_flashcard, t) for t in gen_tasks]), total=len(gen_tasks)):
                    topic, tier, m_key, answer = future.result()
                    if answer:
                        if tier not in cache[topic]["results"][m_key]: cache[topic]["results"][m_key][tier] = {}
                        cache[topic]["results"][m_key][tier]["answer"] = answer
                        save_cache_safely(cache)
            unload_model(actual_tag) 

    print(f"\n⚖️  STAGE 3: Scoring Distractor Noise Adoption [{MEDGEMMA_MODEL}]...")
    eval_tasks = []
    for topic, data in cache.items():
        for model_key, tiers_data in data["results"].items():
            for tier, result in tiers_data.items():
                if result.get("answer") and "adopted_distractor" not in result:
                    eval_tasks.append((
                        topic, 
                        data["distractor_topic"], 
                        data["distractor_key_features"], 
                        tier, 
                        result["answer"], 
                        model_key
                    ))
    
    if eval_tasks:
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            for future in tqdm(concurrent.futures.as_completed([executor.submit(worker_score_adoption, t) for t in eval_tasks]), total=len(eval_tasks)):
                topic, tier, m_key, adopted = future.result()
                if adopted is not None:
                    cache[topic]["results"][m_key][tier]["adopted_distractor"] = adopted
                    save_cache_safely(cache)
        unload_model(MEDGEMMA_MODEL)

    print("\n=========================================================")
    print("   PHASE 37: DISTRACTOR RAG NOISE ADOPTION RATE")
    print("=========================================================\n")
    print("* Lower % means BETTER noise rejection (the model ignored irrelevant retrieved context).\n")
    
    report = "=========================================================\n"
    report += "   PHASE 37: DISTRACTOR RAG NOISE ADOPTION RATE\n"
    report += "=========================================================\n\n"
    report += "* Lower % means BETTER noise rejection.\n\n"
    
    for model_key in MODELS_TO_TEST.keys():
        report += f"--- {model_key} ---\n"
        total_adoptions, total_evals = 0, 0
        
        for tier in TIERS:
            tier_adoptions = sum(1 for data in cache.values() if data["results"].get(model_key, {}).get(tier, {}).get("adopted_distractor", -1) == 1)
            tier_evals = sum(1 for data in cache.values() if "adopted_distractor" in data["results"].get(model_key, {}).get(tier, {}))
            total_adoptions += tier_adoptions
            total_evals += tier_evals
            
            if tier_evals > 0:
                rate = (tier_adoptions / tier_evals) * 100
                report += f"  > {tier.capitalize():<12} : {rate:>5.1f}% Noise Adoption ({tier_adoptions}/{tier_evals})\n"
                
        if total_evals > 0:
            overall_rate = (total_adoptions / total_evals) * 100
            report += f"  > OVERALL      : {overall_rate:>5.1f}% Noise Adoption ({total_adoptions}/{total_evals})\n\n"
        
    print(report)
    with open(REPORT_FILE, "w", encoding="utf-8") as f: f.write(report)

if __name__ == "__main__":
    main()
