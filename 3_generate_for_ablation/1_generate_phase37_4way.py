import os
import chromadb
import requests
import json
import concurrent.futures
import time
import threading
import string
import random
import numpy as np
import torch
from sentence_transformers import CrossEncoder
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from tqdm import tqdm
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)

OLLAMA_CHAT_ENDPOINT = "http://127.0.0.1:11434/api/chat"
OLLAMA_EMBED_ENDPOINT = "http://127.0.0.1:11434/api/embeddings"

DATA_PATH = "phase37_available_eval_pool.jsonl"
USED_TOPICS_FILE = "phase37_used_seed_topics.jsonl"
OUTPUT_FILE = "phase37_4way_multiseed_results.jsonl"

CACHE_CHROMA = "phase37_eval_cache_chroma.jsonl"
CACHE_BASELINE = "phase37_eval_cache_baseline.jsonl"
CACHE_SFT = "phase37_eval_cache_sft.jsonl"

BASELINE_MODEL = "qwen2.5:14b"
SFT_MODEL = "pathology_tutor_phase37_sft:latest"  # Adjust to your Ollama model name
EMBEDDING_MODEL = "bge-large"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

SEEDS_TO_RUN = [40, 50, 99, 123, 456]
file_lock = threading.Lock()
reranker_lock = threading.Lock()

class LocalOllamaEmbeddings(EmbeddingFunction):
    def __init__(self, endpoint, model_name):
        self.endpoint, self.model_name = endpoint, model_name
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for doc in input:
            for attempt in range(3):
                try:
                    res = requests.post(self.endpoint, json={"model": self.model_name, "prompt": doc}, timeout=120)
                    embeddings.append(res.json()["embedding"])
                    break
                except:
                    if attempt == 2: embeddings.append([0.0] * 1024)
                    time.sleep(1)
        return embeddings

def get_production_system_prompt(tier):
    if tier == "foundational":
        return "You are a Foundational Pathology Tutor for Year-1 medical students. Your sole purpose is to test basic rote recall, simple definitions, and morphological features. You are FORBIDDEN from discussing multi-disease comparisons, clinical treatments, or advanced molecular pathways."
    elif tier == "intermediate":
        return "You are a Clinical Pathology Instructor focusing on diagnostic application for residents. Your goal is to test the user's ability to contrast distinct pathological entities, interpret diagnostic criteria, and evaluate multi-marker IHC panels."
    elif tier == "advanced":
        return "You are an Elite Molecular Pathologist and Oncologist. Your mandate is exhaustive clinical synthesis. You must evaluate complex genetic markers, dictate targeted oncology treatments, and predict survival outcomes."

def call_ollama(system_prompt, user_prompt, model_name, seed=None, max_tokens=800):
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.0, "top_p": 0.9, "num_predict": max_tokens, "num_ctx": 4096}
    }
    if seed: payload["options"]["seed"] = seed
    for attempt in range(3):
        try:
            res = requests.post(OLLAMA_CHAT_ENDPOINT, json=payload, timeout=300)
            return res.json().get("message", {}).get("content", "").strip()
        except:
            if attempt == 2: return ""
            time.sleep(2)

def normalize_string(s):
    if not s: return ""
    return s.lower().translate(str.maketrans("", "", string.punctuation)).replace(" ", "")

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

def fetch_and_rerank_context(topic, collection, embedding_function, local_reranker):
    for attempt in range(3):
        try:
            query_vector = embedding_function([topic])[0]
            if sum(query_vector) == 0.0: return ""
            results = collection.query(query_embeddings=[query_vector], n_results=10)
            docs = results.get("documents", [[]])[0]
            if not docs: return ""
            with reranker_lock:
                sentence_pairs = [[topic, doc] for doc in docs]
                scores = local_reranker.predict(sentence_pairs)
            ranked_indices = np.argsort(scores)[::-1]
            best_docs = [docs[i] for i in ranked_indices[:2]]
            return "\n\n".join(best_docs)
        except: time.sleep(2)
    return ""

def main():
    print("🛡️  Loading training topics to enforce strict Zero Data Leakage...")
    used_topics_normalized = set()
    if os.path.exists(USED_TOPICS_FILE):
        with open(USED_TOPICS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                topic = json.loads(line).get("topic", "")
                if topic: used_topics_normalized.add(normalize_string(topic))

    print(f"📚 Scanning {DATA_PATH} for pristine hold-out topics...")
    all_available_topics = []
    seen_eval_topics = set()  # Tracks internal duplicates within the available pool
    
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            topic = json.loads(line).get("topic", "").strip()
            norm_topic = normalize_string(topic)
            
            # Skip if empty, or if it was used in training
            if not topic or norm_topic in used_topics_normalized: 
                continue
                
            # Skip if it is a duplicate inside the hold-out pool itself
            if norm_topic not in seen_eval_topics:
                seen_eval_topics.add(norm_topic)
                all_available_topics.append(topic)

    tasks = []
    unique_topics_to_generate = set()
    for seed in SEEDS_TO_RUN:
        random.seed(seed)
        # Safely samples exactly 50 unique topics per seed
        seed_topics = random.sample(all_available_topics, min(50, len(all_available_topics)))
        for topic in seed_topics:
            unique_topics_to_generate.add(topic)
            for tier in ["foundational", "intermediate", "advanced"]:
                tasks.append({"task_id": f"{seed}_{topic}_{tier}", "topic": topic, "tier": tier, "seed": seed})

    print("\n" + "=" * 50 + f"\nSTAGE 1: GENERATE OPTIMAL CHROMA CONTEXT\n" + "=" * 50)
    chroma_cache = load_cache(CACHE_CHROMA)
    pending_chroma = [t for t in unique_topics_to_generate if t not in chroma_cache]
    
    if pending_chroma:
        print("📂 Connecting to persistent ChromaDB...")
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        embedding_function = LocalOllamaEmbeddings(OLLAMA_EMBED_ENDPOINT, EMBEDDING_MODEL)
        collection = chroma_client.get_collection(name="pathology_textbooks")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        reranker = CrossEncoder(RERANKER_MODEL, max_length=512, device=device)
        
        for t in tqdm(pending_chroma, desc="ChromaDB Fetching"):
            ctx = fetch_and_rerank_context(t, collection, embedding_function, reranker)
            append_to_cache(CACHE_CHROMA, {"id": t, "content": ctx})
        del reranker
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    chroma_cache = load_cache(CACHE_CHROMA)
    for task in tasks: 
        task["context_open"] = chroma_cache.get(task["topic"], "")

    print(f"\n🚀 Running {len(tasks)} evaluations across 4 pipelines...")

    def run_pipeline(task, model_name, cache_file, cache_dict):
        if task["task_id"] in cache_dict: return
        sys_p = get_production_system_prompt(task["tier"])
        
        ans_closed = call_ollama(sys_p, f"Topic Concepts: {task['topic']}\nWrite the clinical question and answer now.", model_name, task["seed"])
        ans_open = call_ollama(sys_p, f"Textbook Context Base:\n{task['context_open']}\n\nTopic Concepts: {task['topic']}\nWrite the clinical question and answer now.", model_name, task["seed"])
        
        append_to_cache(cache_file, {"id": task["task_id"], "content": {"closed": ans_closed, "open": ans_open}})

    for model, c_file, desc in [(BASELINE_MODEL, CACHE_BASELINE, "Baseline"), (SFT_MODEL, CACHE_SFT, "Phase 37 SFT")]:
        cache_data = load_cache(c_file)
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futs = [executor.submit(run_pipeline, t, model, c_file, cache_data) for t in tasks]
            for _ in tqdm(concurrent.futures.as_completed(futs), total=len(tasks), desc=f"{desc} Pass"): pass

    print(f"\n💾 Merging caches and writing final matrix to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        grouped = {}
        for task in tasks:
            key = (task["topic"], task["seed"])
            if key not in grouped: grouped[key] = {}
            tid = task["task_id"]
            grouped[key][task["tier"]] = {
                "p1_baseline_closed": load_cache(CACHE_BASELINE).get(tid, {}).get("closed", ""),
                "p2_baseline_open": load_cache(CACHE_BASELINE).get(tid, {}).get("open", ""),
                "p3_sft_closed": load_cache(CACHE_SFT).get(tid, {}).get("closed", ""),
                "p4_sft_open": load_cache(CACHE_SFT).get(tid, {}).get("open", "")
            }
        
        for (topic, seed), tiers_data in grouped.items():
            f.write(json.dumps({
                "run_seed": seed, "seed_topic": topic, "pipeline_outputs": tiers_data
            }) + "\n")
    print("✅ Finished Generation!")

if __name__ == "__main__":
    main()
