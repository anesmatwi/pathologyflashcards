import os
import glob
import json
import requests
import time
import re
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIGURATION ---
INPUT_FOLDER = "./processed_txt_v2"
OUTPUT_FILE = "./train_data.jsonl"
TRACKER_FILE = "./processed_books.log" 

WORKER_NODES = [
    "http://192.168.0.111:80/api/generate",
    "http://192.168.0.100:80/api/generate"
]
node_cycle = itertools.cycle(WORKER_NODES)

MODEL = "hf.co/bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF:Q6_K" 
MAX_WORKERS = 2 

def query_deepseek(prompt):
    payload = {
        "model": MODEL, 
        "prompt": prompt, 
        "stream": False, 
        "options": {"temperature": 0.3, "num_ctx": 8192, "num_predict": 2000} 
    }
    for _ in range(3):
        try:
            target_node = next(node_cycle)
            res = requests.post(target_node, json=payload, timeout=240)
            if res.status_code == 200:
                text = res.json().get('response', '')
                return text.split("</think>")[-1].strip() if "</think>" in text else text.strip()
        except Exception as e: 
            time.sleep(2)
    return None

def process_text_chunk(chunk_text, chunk_id):
    prompt = f"""You are an expert medical educator developing an adaptive pathology curriculum. 
Read the following textbook excerpt and generate exactly THREE flashcards based on Bloom's Taxonomy.

RULES:
1. Beginner (Remember): Use verbs like define, list, or identify.
2. Intermediate (Understand/Apply): Use verbs like explain, summarize, apply, or demonstrate.
3. Advanced (Analyze/Evaluate): Use verbs like compare, differentiate, evaluate, justify, or assess.
4. The "answer" field MUST be highly expansive. You must explicitly structure your answer to include three distinct elements: (1) The core pathophysiological mechanism, (2) Direct clinical correlations or manifestations, and (3) A brief explanation of why related alternative concepts would be incorrect.
5. Ensure absolutely no hallucinations. All facts must be grounded directly in the provided text.
6. If the provided text is a Table of Contents, Index, Reference List, or lacks substantive medical knowledge, output an empty JSON array: []

OUTPUT FORMAT:
Provide the output strictly as a JSON array of objects with keys: "difficulty", "question", "answer", "search_index". 
The "search_index" must be an array of 3-5 highly specific clinical keywords.
Do not include markdown formatting like ```json.

EXAMPLE GOLD-STANDARD OUTPUT:
[
  {{
    "difficulty": "Advanced",
    "question": "Evaluate the role of prompt reperfusion in a patient presenting with an acute ST-segment elevation myocardial infarction (STEMI) and justify how this necessary intervention can paradoxically exacerbate cellular injury.",
    "answer": "Prompt reperfusion is the definitive treatment for a STEMI to restore blood flow and halt ischemic coagulative necrosis. (1) Pathophysiologically, this intervention can paradoxically cause reperfusion injury because the sudden reintroduction of oxygen to ischemic tissue generates massive amounts of reactive oxygen species (ROS) from damaged mitochondria, while the sudden influx of calcium into calcium-depleted cells causes hypercontraction (contraction band necrosis). (2) Clinically, this manifests as arrhythmias or depressed myocardial function (stunning) immediately following PCI or thrombolytic therapy. (3) While one might assume ischemia alone is the sole driver of infarct size, this is incorrect; the biochemical shock of reperfusion itself independently exacerbates localized cellular damage, making adjunctive therapies to limit ROS crucial.",
    "search_index": ["Myocardial Infarction", "Reperfusion Injury", "Coagulative Necrosis", "Reactive Oxygen Species", "Contraction Band Necrosis"]
  }}
]

TEXT EXCERPT:
{chunk_text}
"""
    
    result = query_deepseek(prompt)
    if not result or result == "[]":
        print(f"    Chunk {chunk_id} Skipped: Deemed irrelevant or missing context.")
        return []

    clean_json = re.sub(r'^```json\s*|\s*```$', '', result, flags=re.MULTILINE).strip()
    
    try:
        return json.loads(clean_json)
    except json.JSONDecodeError:
        print(f"  [!] JSON Parse Error on chunk {chunk_id}. Raw output snippet: {clean_json[:100]}...")
        return []

def load_processed_files():
    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f)
    return set()

def mark_as_processed(filename):
    with open(TRACKER_FILE, "a", encoding="utf-8") as f:
        f.write(filename + "\n")

def main():
    if not os.path.exists(INPUT_FOLDER):
        print(f"Folder {INPUT_FOLDER} not found.")
        return

    files = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.txt")))
    processed_files = load_processed_files()
    print(f"Found {len(files)} total text files. ({len(processed_files)} already processed).")

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out_file:
        for file_path in files:
            filename = os.path.basename(file_path)
            
            if filename in processed_files:
                print(f"⏭️ Skipping '{filename}' (Already fully processed).")
                continue
                
            print(f"\n-> Generating flashcards for: {filename}")
            
            with open(file_path, "r", encoding="utf-8") as f:
                full_text = f.read()

            # --- THE OVERLAPPING CHUNK LOGIC ---
            words = full_text.split()
            chunk_size = 1500
            overlap = 200
            step = chunk_size - overlap
            
            chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), max(1, step))]

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {executor.submit(process_text_chunk, chunk, i): i for i, chunk in enumerate(chunks)}
                
                for future in as_completed(futures):
                    chunk_id = futures[future]
                    try:
                        flashcards = future.result()
                        if flashcards:
                            for card in flashcards:
                                card['source_document'] = filename 
                                out_file.write(json.dumps(card) + "\n")
                            print(f"    Chunk {chunk_id} Done: Saved {len(flashcards)} cards.")
                    except Exception as e:
                        print(f"    [!] Thread Error on chunk {chunk_id}: {e}")
            
            mark_as_processed(filename)

if __name__ == "__main__":
    main()
