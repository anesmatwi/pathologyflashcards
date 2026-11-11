import os
import json
import random
import threading
import time
import requests
import concurrent.futures
import numpy as np
import torch
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import CrossEncoder
from tqdm import tqdm
import logging
import itertools
import string

logging.getLogger("httpx").setLevel(logging.WARNING)

# ==========================================
# 1. CONFIGURATION
# ==========================================
OLLAMA_CHAT_ENDPOINT = "http://127.0.0.1:11434/api/chat"
OLLAMA_EMBED_ENDPOINT = "http://127.0.0.1:11434/api/embeddings"

# Set these to the exact models you are evaluating
BASELINE_MODEL = "qwen2.5:14b"
SFT_MODEL = "pathology_tutor_phase37_sft:latest" 

EMBEDDING_MODEL = "bge-large"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
MAX_WORKERS = 6
EVAL_TOPIC_COUNT = 100

EVAL_POOL_MANIFEST = "phase37_available_eval_pool.jsonl"
USED_TOPICS_FILE = "phase37_used_seed_topics.jsonl"
OUTPUT_FILE = "phase37_distractor_generations_100.jsonl"

file_lock = threading.Lock()
reranker_lock = threading.Lock()

SYS_PROMPTS = {
    "foundational": "You are a Year-1 Foundational Pathology Tutor.\nRULES:\n1. Answer MUST be BRIEF and CONCISE (under 45 words).\n2. Rote recall only. NO genetics.\n3. Filter out any advanced genetics or treatments. Extract ONLY the basic definitions.\nRAG MODE: Use the provided Textbook Context to ground your answer.\nFORMATTING: You MUST format your output strictly using these tags:\n[QUESTION] <write question stem>\n[ANSWER] <write answer>",
    "intermediate": "You are a Clinical Pathology Instructor for residents.\nRULES:\n1. Answer MUST be detailed but focused (strictly between 60 and 110 words).\n2. Diagnostic criteria or multi-marker IHC.\n3. NO complex genetics.\nRAG MODE: Use the provided Textbook Context to ground your answer.\nFORMATTING: You MUST format your output strictly using these tags:\n[QUESTION] <write question stem>\n[ANSWER] <write answer>",
    "advanced": "You are an Elite Molecular Pathologist.\nRULES:\n1. Answer MUST be a LONG, EXHAUSTIVE explanation (strictly AT LEAST 150 words).\n2. Exhaustive genetics/targeted oncology.\nRAG MODE: Use the provided Textbook Context to ground your answer.\nFORMATTING: You MUST format your output strictly using these tags:\n[QUESTION] <write question stem>\n[ANSWER] <write answer>"
}

# ==========================================
# 2. HELPER FUNCTIONS & CHROMADB CLASSES
# ==========================================
def normalize_string(s):
    if not s: return ""
    return s.lower().translate(str.maketrans("", "", string.punctuation)).replace(" ", "")

class LocalOllamaEmbeddings(EmbeddingFunction):
    def __init__(self, endpoint, model_name):
        self.endpoint, self.model_name = endpoint, model_name
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for doc in input:
            for _ in range(3):
                try:
                    res = requests.post(self.endpoint, json={"model": self.model_name, "prompt": doc, "keep_alive": -1}, timeout=120)
                    embeddings.append(res.json()["embedding"])
                    break
                except: time.sleep(1)
        return embeddings

def fetch_and_rerank_context(topic, reranker_cycle, collection, embedding_function):
    for _ in range(3):
        try:
            query_vector = embedding_function([topic])[0]
            if sum(query_vector) == 0.0: return []
            
            docs = collection.query(query_embeddings=[query_vector], n_results=10).get("documents", [[]])[0]
            if not docs: return []
            
            with reranker_lock: local_reranker = next(reranker_cycle)
            scores = local_reranker.predict([[topic, doc] for doc in docs])
            
            top_indices = np.argsort(scores)[::-1][:5]
            return [docs[i] for i in top_indices]
        except Exception:
            time.sleep(2)
    return []

# ==========================================
# 3. GENERATION PIPELINE
# ==========================================
def call_ollama(model, sys_prompt, user_prompt, max_tokens):
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
        "stream": False, "keep_alive": -1,
        "options": {"temperature": 0.2, "num_predict": max_tokens, "num_ctx": 4096}
    }
    for _ in range(3):
        try:
            res = requests.post(OLLAMA_CHAT_ENDPOINT, json=payload, timeout=120)
            return res.json().get("message", {}).get("content", "").strip()
        except: time.sleep(2)
    return None

def process_generation(task):
    target_topic = task["target_topic"]
    distractor_topic = task["distractor_topic"]
    distractor_chunks = task["distractor_chunks"]
    tier = task["tier"]
    
    context_text = "\n\n".join(distractor_chunks[:3]) 
    user_prompt = f"Textbook Context Base:\n{context_text}\n\nTopic: {target_topic}\nWrite the clinical question and answer now."
    sys_prompt = SYS_PROMPTS[tier]
    
    max_tokens = 250 if tier == "foundational" else 450 if tier == "intermediate" else 1000

    # 1. Baseline Generation
    base_text = call_ollama(BASELINE_MODEL, sys_prompt, user_prompt, max_tokens)
    if base_text:
        with file_lock:
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "pipeline": "Baseline_Qwen", "tier": tier, 
                    "target_topic": target_topic, "distractor_topic": distractor_topic, "text": base_text
                }) + "\n")

    # 2. SFT Generation
    sft_text = call_ollama(SFT_MODEL, sys_prompt, user_prompt, max_tokens)
    if sft_text:
        with file_lock:
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "pipeline": "Phase37_SFT", "tier": tier, 
                    "target_topic": target_topic, "distractor_topic": distractor_topic, "text": sft_text
                }) + "\n")

# ==========================================
# 4. ORCHESTRATION
# ==========================================
def main():
    if not os.path.exists(EVAL_POOL_MANIFEST):
        print(f"❌ {EVAL_POOL_MANIFEST} not found. Ensure holdout data exists.")
        return

    print("🛡️ Loading training topics to enforce strict Zero Data Leakage...")
    used_topics_normalized = set()
    if os.path.exists(USED_TOPICS_FILE):
        with open(USED_TOPICS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                topic = json.loads(line).get("topic", "")
                if topic: used_topics_normalized.add(normalize_string(topic))

    print(f"📚 Scanning {EVAL_POOL_MANIFEST} for pristine evaluation topics...")
    eval_topics = []
    seen_eval_topics = set()
    with open(EVAL_POOL_MANIFEST, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            topic = json.loads(line).get("topic", "").strip()
            norm_topic = normalize_string(topic)
            
            if not topic or norm_topic in used_topics_normalized:
                continue
                
            if norm_topic not in seen_eval_topics:
                seen_eval_topics.add(norm_topic)
                eval_topics.append(topic)
                
            if len(eval_topics) >= EVAL_TOPIC_COUNT: break

    print(f"📚 Retrieving ChromaDB contexts for {len(eval_topics)} unique evaluation topics...")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    embedding_function = LocalOllamaEmbeddings(OLLAMA_EMBED_ENDPOINT, EMBEDDING_MODEL)
    collection = chroma_client.get_collection(name="pathology_textbooks")
    
    gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    rerankers = [CrossEncoder(RERANKER_MODEL, max_length=512, device=f'cuda:{i}' if torch.cuda.is_available() else 'cpu') for i in range(gpus)]
    
    topic_contexts = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=gpus * 2) as exc:
        futs = {exc.submit(fetch_and_rerank_context, t, itertools.cycle(rerankers), collection, embedding_function): t for t in eval_topics}
        for future in tqdm(concurrent.futures.as_completed(futs), total=len(eval_topics), desc="Retrieving Truth Contexts"):
            topic_contexts.append({"topic": futs[future], "chunks": future.result()})
            
    # Free VRAM
    del rerankers
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Apply the Matrix Shift (Topic i gets chunks from Topic i+1)
    tasks = []
    for i in range(len(topic_contexts)):
        target = topic_contexts[i]["topic"]
        distractor_idx = (i + 1) % len(topic_contexts) # Wrap around at the end
        
        distractor_topic = topic_contexts[distractor_idx]["topic"]
        distractor_chunks = topic_contexts[distractor_idx]["chunks"]
        
        if not distractor_chunks: continue # Skip if Chroma failed to find chunks for distractor
        
        for tier in ["foundational", "intermediate", "advanced"]:
            tasks.append({
                "target_topic": target,
                "distractor_topic": distractor_topic,
                "distractor_chunks": distractor_chunks,
                "tier": tier
            })

    print(f"🚀 Prepared {len(tasks)} target/distractor task generations.")
    open(OUTPUT_FILE, 'w').close() # Reset file
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as exc:
        futs = [exc.submit(process_generation, t) for t in tasks]
        for _ in tqdm(concurrent.futures.as_completed(futs), total=len(tasks), desc="Generating Mismatched Cards"):
            pass

    print(f"✅ Generated 600 mismatched cards! Run subsequent scoring script next.")

if __name__ == "__main__":
    main()
