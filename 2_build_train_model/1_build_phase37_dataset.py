import os
import json
import random
import re
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

# Configure Hard Error Logging
logging.basicConfig(
    filename='phase37_error.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ==========================================
# 1. CONFIGURATION & FILE MANIFESTS
# ==========================================
OLLAMA_CHAT_ENDPOINT = "http://127.0.0.1:11434/api/chat"
OLLAMA_GENERATE_ENDPOINT = "http://127.0.0.1:11434/api/generate"
OLLAMA_EMBED_ENDPOINT = "http://127.0.0.1:11434/api/embeddings"

# Models
GENERATOR_MODEL = "command-r:35b"
KNOWLEDGE_MODEL = "medgemma:27b"
EMBEDDING_MODEL = "bge-large"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
MAX_WORKERS = 6

# Data Targets
TARGET_SFT_TOTAL = 7500
TARGET_SFT_MULTI = 2250

# Files
SEED_VAULT = "train_data.jsonl"
EVAL_POOL_MANIFEST = "phase37_available_eval_pool.jsonl"
USED_TOPICS_MANIFEST = "phase37_used_seed_topics.jsonl"

# Checkpoints
CONTEXT_CACHE = "phase37_context_cache.jsonl"
MEDGEMMA_CACHE = "phase37_medgemma_cache.jsonl"
OUT_SFT = "phase37_1_sft_dataset.jsonl"

file_lock = threading.Lock()
reranker_lock = threading.Lock()
context_lock = threading.Lock()
medgemma_lock = threading.Lock()
manifest_lock = threading.Lock()

# ==========================================
# 2. SYSTEM PROMPTS (SFT Training Triggers)
# ==========================================
SYS_PROMPTS = {
    "foundational": "You are a Foundational Pathology Tutor for Year-1 medical students. Your sole purpose is to test basic rote recall, simple definitions, and morphological features. You are FORBIDDEN from discussing multi-disease comparisons, clinical treatments, or advanced molecular pathways.",
    "intermediate": "You are a Clinical Pathology Instructor focusing on diagnostic application for residents. Your goal is to test the user's ability to contrast distinct pathological entities, interpret diagnostic criteria, and evaluate multi-marker IHC panels.",
    "advanced": "You are an Elite Molecular Pathologist and Oncologist. Your mandate is exhaustive clinical synthesis. You must evaluate complex genetic markers, dictate targeted oncology treatments, and predict survival outcomes."
}

# ==========================================
# 3. CONTEXT RETRIEVAL & CACHING
# ==========================================
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
        except Exception as e:
            time.sleep(2)
    logging.error(f"ChromaDB retrieval failed for topic: {topic}")
    return []

def fetch_and_cache_context(topic, reranker_cycle, collection, embedding_function):
    chunks = fetch_and_rerank_context(topic, reranker_cycle, collection, embedding_function)
    with context_lock:
        with open(CONTEXT_CACHE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"topic": topic, "chunks": chunks}) + "\n")
    return chunks

# ==========================================
# 4. SFT CONTEXT PROFILING (Noise Injection)
# ==========================================
def build_sft_user_prompt(topic, target_chunks, all_cached_contexts):
    rand = random.random()
    context_text = ""
    
    if rand < 0.35:
        context_text = "\n\n".join(target_chunks[:2])
    elif rand < 0.60:
        num_chunks = random.choice([4, 5])
        context_text = "\n\n".join(target_chunks[:num_chunks])
    elif rand < 0.75:
        # Distractor Safety: Ensure the chosen topic actually has chunks
        unrelated_topic = random.choice(list(all_cached_contexts.keys()))
        while unrelated_topic == topic or len(all_cached_contexts.get(unrelated_topic, [])) == 0:
            unrelated_topic = random.choice(list(all_cached_contexts.keys()))
        
        distractor_chunks = all_cached_contexts[unrelated_topic]
        num_chunks = random.choice([2, 3, 4])
        context_text = "\n\n".join(distractor_chunks[:num_chunks])
    elif rand < 0.85:
        if target_chunks:
            context_text = target_chunks[0][:random.randint(120, 200)] + "..."

    if context_text.strip():
        return f"Textbook Context Base:\n{context_text.strip()}\n\nTopic Concepts: {topic}\nWrite the clinical question and answer now."
    else:
        return f"Topic Concepts: {topic}\nWrite the clinical question and answer now."

# ==========================================
# 5. LLM API, GENERATORS, & AUDITOR
# ==========================================
def unload_model(model_name):
    try:
        requests.post(OLLAMA_GENERATE_ENDPOINT, json={"model": model_name, "keep_alive": 0}, timeout=10)
        print(f"🧹 Unloaded {model_name} from VRAM.")
    except: pass

def call_ollama(model, sys_prompt, user_prompt, max_tokens):
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
        "stream": False, "keep_alive": -1,
        "options": {"temperature": 0.2, "num_predict": max_tokens, "num_ctx": 8192, "num_gpu": 99}
    }
    for _ in range(3):
        try:
            res = requests.post(OLLAMA_CHAT_ENDPOINT, json=payload, timeout=180)
            return res.json().get("message", {}).get("content", "").strip()
        except Exception as e: time.sleep(2)
    return None

def generate_medgemma_brief(topic):
    sys_prompt = (
        "You are an Elite Molecular Pathologist. Output ONLY raw, highly exhaustive medical facts. "
        "DO NOT write questions or introductory filler. You MUST keep your response strictly under 350 words."
    )
    user_prompt = (
        f"Write an ultra-concise, bulleted medical brief about {topic}. "
        "Focus entirely on molecular pathways, targeted oncology, immunohistochemistry, and complex genetics. "
        "Use short, fragmented sentences to maximize data density."
    )
    
    payload = {
        "model": KNOWLEDGE_MODEL,
        "prompt": f"{sys_prompt}\n\n{user_prompt}",
        "stream": False, "keep_alive": -1,
        "options": {"temperature": 0.2, "num_predict": 500, "num_ctx": 2048, "num_gpu": 99}
    }
    for _ in range(3):
        try:
            res = requests.post(OLLAMA_GENERATE_ENDPOINT, json=payload, timeout=120)
            return res.json().get("response", "").strip()
        except: time.sleep(2)
    logging.error(f"MedGemma failed to generate brief for: {topic}")
    return ""

def process_medgemma_task(topic):
    brief = generate_medgemma_brief(topic)
    if brief:
        with medgemma_lock:
            with open(MEDGEMMA_CACHE, "a", encoding="utf-8") as f:
                f.write(json.dumps({"topic": topic, "brief": brief}) + "\n")
    return bool(brief)

def extract_qa(raw_text):
    if not raw_text: return None, None
    try:
        c = re.sub(r"\*\*\[?(QUESTION|ANSWER)\]?\*\*:?", r"[\1]", raw_text, flags=re.IGNORECASE)
        if c.count("[QUESTION]") > 1 or c.count("[ANSWER]") > 1: return None, None
        q = re.search(r"\[QUESTION\]\s*(.*?)\s*\[ANSWER\]", c, re.DOTALL | re.IGNORECASE)
        a = re.search(r"\[ANSWER\]\s*(.*?)(?:\[/ANSWER\]|$)", c, re.DOTALL | re.IGNORECASE)
        bad = ["create a", "here is your", "front:", "back:", "flashcard:"]
        if not q or not a or any(b in q.group(1).lower() for b in bad): return None, None
        return q.group(1).strip(), a.group(1).strip()
    except: return None, None

def check_length(tier, text):
    wc = len(text.split())
    if tier == "foundational": return wc < 45
    elif tier == "intermediate": return 60 <= wc <= 110
    elif tier == "advanced": return wc >= 150
    return False

def audit_row(seed_topic, generated_question):
    audit_prompt = f"""You are an expert Medical Data Auditor. Your task is to detect "RAG Hijacking".
REQUESTED SEED TOPIC: {seed_topic}
GENERATED QUESTION: {generated_question}
Did the generated question accurately target the requested seed topic, or was it hijacked by unrelated topic concepts?
Output ONLY a JSON object.
JSON Schema:
{{
  "is_hijacked": <boolean true or false>
}}"""
    payload = {
        "model": GENERATOR_MODEL, "prompt": audit_prompt, "stream": False, "format": "json", "keep_alive": -1,
        "options": {"temperature": 0.0, "num_predict": 128, "num_gpu": 99}
    }
    for _ in range(3):
        try:
            res = requests.post(OLLAMA_GENERATE_ENDPOINT, json=payload, timeout=60)
            return json.loads(res.json().get("response", "{}")).get("is_hijacked", False)
        except: time.sleep(1)
    return False

def get_generator_sys_prompt(tier, is_rag=True):
    factuality_mandate = "FACTUALITY MANDATE: All IHC markers, genetic mutations, staining patterns, and drug names must strictly align with current WHO classification guidelines. Never invent false markers."
    anti_leak = "ANTI-LEAK: DO NOT output conversational filler. DO NOT write patient case vignettes. Ask direct, conceptual questions only."
    formatting = "FORMATTING: You MUST format your output strictly using these tags:\n[QUESTION] <write question stem>\n[ANSWER] <write answer>"
    rag_rule = "RAG MODE: Use the provided Textbook Context to ground your answer." if is_rag else "CLOSED BOOK MODE: Rely purely on your internal clinical knowledge."
    
    if tier == "foundational":
        return f"You are a Year-1 Foundational Pathology Tutor.\nRULES:\n1. Answer MUST be BRIEF and CONCISE (under 45 words).\n2. Rote recall only. NO genetics.\n3. Filter out any advanced genetics or treatments. Extract ONLY the basic definitions.\n{rag_rule}\n{factuality_mandate}\n{anti_leak}\n{formatting}"
    elif tier == "intermediate":
        return f"You are a Clinical Pathology Instructor for residents.\nRULES:\n1. Answer MUST be detailed but focused (strictly between 60 and 110 words).\n2. Diagnostic criteria or multi-marker IHC.\n3. NO complex genetics.\n{rag_rule}\n{factuality_mandate}\n{anti_leak}\n{formatting}"
    elif tier == "advanced":
        return f"You are an Elite Molecular Pathologist.\nRULES:\n1. Answer MUST be a LONG, EXHAUSTIVE explanation (strictly AT LEAST 150 words).\n2. Exhaustive genetics/targeted oncology.\n3. Use 'Curated Core Facts' as a factual floor and heavily expand upon them.\n{rag_rule}\n{factuality_mandate}\n{anti_leak}\n{formatting}"

def generate_clean_qa(tier, topic, clean_chunks, curated_facts, max_tokens):
    for _ in range(3): 
        sys_prompt = get_generator_sys_prompt(tier, is_rag=True)
        
        parts = []
        if clean_chunks: parts.append(f"Textbook Context Base:\n{chr(10).join(clean_chunks[:3])}\n")
        if curated_facts: parts.append(f"Curated Core Facts (Extremely Important):\n{curated_facts}\n")
        parts.append(f"Topic: {topic}\nWrite the clinical question and answer now.")
        
        user_prompt = "\n".join(parts)
        
        raw_text = call_ollama(GENERATOR_MODEL, sys_prompt, user_prompt, max_tokens)
        q, a = extract_qa(raw_text)
        
        if not q or not a or not check_length(tier, a): 
            continue
        
        if clean_chunks and audit_row(topic, q):
            sys_cb = get_generator_sys_prompt(tier, is_rag=False)
            user_cb = f"Curated Core Facts:\n{curated_facts}\n\nTopic: {topic}\nWrite the clinical question and answer now." if curated_facts else f"Topic: {topic}\nWrite the clinical question and answer now."
            
            raw_cb = call_ollama(GENERATOR_MODEL, sys_cb, user_cb, max_tokens)
            q_cb, a_cb = extract_qa(raw_cb)
            
            if q_cb and a_cb and check_length(tier, a_cb): return q_cb, a_cb
            continue
            
        return q, a
    logging.error(f"Failed to generate valid QA for tier: {tier} | topic: {topic}")
    return None, None

def process_sft_task(topic, is_multi_tier, chunks, medgemma_brief, curated_seed_facts, all_contexts_dict):
    f_q, f_a = generate_clean_qa("foundational", topic, chunks, curated_seed_facts, 250)
    if not f_q or not f_a: return False

    sft_rows = []
    
    f_user_prompt = build_sft_user_prompt(topic, chunks, all_contexts_dict)
    sft_rows.append({
        "seed_topic": topic,
        "messages": [
            {"role": "system", "content": SYS_PROMPTS["foundational"]},
            {"role": "user", "content": f_user_prompt},
            {"role": "assistant", "content": f"[QUESTION] {f_q}\n\n[ANSWER] {f_a}"}
        ]
    })

    if is_multi_tier:
        i_q, i_a = generate_clean_qa("intermediate", topic, chunks, curated_seed_facts, 450)
        a_q, a_a = generate_clean_qa("advanced", topic, chunks, medgemma_brief, 1500)
        
        if not i_q or not a_q: return False
        
        i_user_prompt = build_sft_user_prompt(topic, chunks, all_contexts_dict)
        a_user_prompt = build_sft_user_prompt(topic, chunks, all_contexts_dict)
        
        sft_rows.append({
            "seed_topic": topic,
            "messages": [
                {"role": "system", "content": SYS_PROMPTS["intermediate"]},
                {"role": "user", "content": i_user_prompt},
                {"role": "assistant", "content": f"[QUESTION] {i_q}\n\n[ANSWER] {i_a}"}
            ]
        })
        sft_rows.append({
            "seed_topic": topic,
            "messages": [
                {"role": "system", "content": SYS_PROMPTS["advanced"]},
                {"role": "user", "content": a_user_prompt},
                {"role": "assistant", "content": f"[QUESTION] {a_q}\n\n[ANSWER] {a_a}"}
            ]
        })

    with file_lock:
        with open(OUT_SFT, "a", encoding="utf-8") as f:
            for row in sft_rows:
                f.write(json.dumps(row) + "\n")
                
    with manifest_lock:
        with open(USED_TOPICS_MANIFEST, "a", encoding="utf-8") as mf:
            mf.write(json.dumps({"topic": topic}) + "\n")
            
    return True

# ==========================================
# 6. ORCHESTRATION & RESUMABILITY
# ==========================================
def main():
    if not os.path.exists(SEED_VAULT):
        print(f"❌ Error: {SEED_VAULT} not found.")
        return

    # ==========================================
    # CARVING CARDS & PREVENTING LEAKAGE
    # ==========================================
    source_cards = []
    seen_topics = set()
    raw_topics_data = {} 
    
    with open(SEED_VAULT, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                search_terms = data.get("search_index", [])
                ans = data.get("answer", "").strip()
                fallback_t = (data.get("topic") or data.get("seed_topic") or "").strip()
                
                card_topics = []
                
                if isinstance(search_terms, list) and len(search_terms) > 0:
                    for t in search_terms:
                        clean_t = t.strip()
                        if clean_t and clean_t not in seen_topics:
                            seen_topics.add(clean_t)
                            card_topics.append(clean_t)
                
                if not card_topics and fallback_t and fallback_t not in seen_topics:
                    seen_topics.add(fallback_t)
                    card_topics.append(fallback_t)
                    
                if card_topics:
                    source_cards.append({
                        "topics": card_topics,
                        "answer": ans
                    })
            except Exception: pass

    # Shuffle the source cards, NOT the individual topics
    random.seed(3407)
    random.shuffle(source_cards)
    
    sft_pool = []
    eval_pool = []
    
    for card in source_cards:
        # Populate the lookup dictionary
        for t in card["topics"]:
            raw_topics_data[t] = card["answer"]
            
        # Hard boundary routing: all topics from one card go to the SAME pool
        if len(sft_pool) < TARGET_SFT_TOTAL:
            sft_pool.extend(card["topics"])
        else:
            eval_pool.extend(card["topics"])

    # Trim SFT pool exactly to target. We discard overflow to avoid leakage.
    if len(sft_pool) > TARGET_SFT_TOTAL:
        sft_pool = sft_pool[:TARGET_SFT_TOTAL]

    multi_tier_pool = set(sft_pool[:TARGET_SFT_MULTI])
    
    with open(EVAL_POOL_MANIFEST, "w", encoding="utf-8") as ef:
        for t in eval_pool: 
            ef.write(json.dumps({"topic": t}) + "\n")
    print(f"📋 Exported {len(eval_pool)} pure hold-out topics to '{EVAL_POOL_MANIFEST}'.")

    # ==========================================
    # STAGE 1: CONTEXT RETRIEVAL
    # ==========================================
    print("\n🧠 STAGE 1: FETCHING HIGH-FIDELITY RAG CONTEXTS")
    topic_to_chunks = {}
    if os.path.exists(CONTEXT_CACHE):
        with open(CONTEXT_CACHE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    topic_to_chunks[d["topic"]] = d.get("chunks", [])

    missing_contexts = [t for t in sft_pool if t not in topic_to_chunks]
    if missing_contexts:
        print(f"📚 Retrieving ChromaDB contexts for {len(missing_contexts)} topics...")
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        embedding_function = LocalOllamaEmbeddings(OLLAMA_EMBED_ENDPOINT, EMBEDDING_MODEL)
        collection = chroma_client.get_collection(name="pathology_textbooks")
        
        gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        rerankers = [CrossEncoder(RERANKER_MODEL, max_length=512, device=f'cuda:{i}' if torch.cuda.is_available() else 'cpu') for i in range(gpus)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=gpus * 2) as exc:
            fut = {exc.submit(fetch_and_cache_context, t, itertools.cycle(rerankers), collection, embedding_function): t for t in missing_contexts}
            for future in tqdm(concurrent.futures.as_completed(fut), total=len(missing_contexts), desc="Retrieving Contexts"):
                topic_to_chunks[fut[future]] = future.result()
        del rerankers
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    else:
        print("⚡ All required contexts found in local cache!")

    # ==========================================
    # STAGE 2: MEDGEMMA BATCH GENERATION
    # ==========================================
    print("\n🔬 STAGE 2: GENERATING MEDGEMMA ADVANCED BRIEFS")
    medgemma_briefs = {}
    if os.path.exists(MEDGEMMA_CACHE):
        with open(MEDGEMMA_CACHE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    medgemma_briefs[d["topic"]] = d["brief"]

    missing_medgemma = [t for t in multi_tier_pool if t not in medgemma_briefs]
    if missing_medgemma:
        print(f"🧬 Booting MedGemma 27B for {len(missing_medgemma)} topics...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as exc:
            futs = [exc.submit(process_medgemma_task, t) for t in missing_medgemma]
            for _ in tqdm(concurrent.futures.as_completed(futs), total=len(futs), desc="MedGemma Briefs"): pass
        
        with open(MEDGEMMA_CACHE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    medgemma_briefs[d["topic"]] = d["brief"]
    else:
        print("⚡ All MedGemma briefs found in local cache!")

    unload_model(KNOWLEDGE_MODEL)
    time.sleep(3)

    # ==========================================
    # STAGE 3: COMMAND R SFT SYNTHESIS
    # ==========================================
    print("\n🚀 STAGE 3: SYNTHESIZING FINAL SFT DATASET (COMMAND R 35B)")
    processed_sft_topics = set()
    
    if os.path.exists(USED_TOPICS_MANIFEST):
        with open(USED_TOPICS_MANIFEST, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try: processed_sft_topics.add(json.loads(line).get("topic"))
                    except: pass
                    
    elif os.path.exists(OUT_SFT):
        with open(OUT_SFT, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try: processed_sft_topics.add(json.loads(line).get("seed_topic"))
                    except: pass

    sft_todo = [t for t in sft_pool if t not in processed_sft_topics]
    
    if not sft_todo:
        print("🎉 Target limit reached for SFT Partition! Dataset complete.")
        return

    print(f"🎯 Generating {len(sft_todo)} total remaining topics...")
    tasks = []
    for t in sft_todo:
        is_multi = t in multi_tier_pool
        brief = medgemma_briefs.get(t, "") if is_multi else ""
        seed_facts = raw_topics_data.get(t, "")
        tasks.append((t, is_multi, topic_to_chunks.get(t, []), brief, seed_facts, topic_to_chunks))

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as exc:
        futs = [exc.submit(process_sft_task, *task) for task in tasks]
        for _ in tqdm(concurrent.futures.as_completed(futs), total=len(tasks), desc="SFT Generation"): pass

    print("\n" + "=" * 60)
    print("🎯 PHASE 37 DATASET COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
