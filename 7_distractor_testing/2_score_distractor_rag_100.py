import concurrent.futures
import json
import logging
import os
import re
import threading
from tqdm import tqdm
import requests
import time

logging.basicConfig(level=logging.WARNING)

# ==========================================
# 1. CONFIGURATION
# ==========================================
OLLAMA_ENDPOINT = "http://127.0.0.1:11434/api/generate"
JUDGE_MODEL = "hf.co/rex099/RISE-Judge-Qwen2.5-32B-GGUF:Q4_K_M"

INPUT_FILE = "phase37_distractor_generations_100.jsonl"
OUTPUT_FILE = "phase37_distractor_scored_100.jsonl"
REPORT_FILE = "phase37_distractor_report_100.txt"

file_lock = threading.Lock()


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def call_ollama(prompt):
  payload = {
      "model": JUDGE_MODEL,
      "prompt": prompt,
      "stream": False,
      "format": "json",
      "options": {"temperature": 0.0, "num_predict": 128, "num_gpu": 99},
  }
  for attempt in range(3):
    try:
      # INCREASED TIMEOUT TO 300 SECONDS
      res = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=300)
      res.raise_for_status()
      return res.json().get("response", "").strip()
    except Exception as e:
      logging.error(f"Ollama call failed (Attempt {attempt+1}/3): {e}")
      time.sleep(3)
  return None


def evaluate_noise_adoption(target_topic, distractor_topic, text):
  prompt = f"""You are an elite Clinical Auditor detecting AI hallucinations and Context Sycophancy.
The AI was instructed to write a medical flashcard focused STRICTLY on the TARGET TOPIC: "{target_topic}".
However, it was maliciously provided with irrelevant background text about the DISTRACTOR TOPIC: "{distractor_topic}".

Read the AI's generated flashcard below.
Did the AI adopt the distractor noise?
- Answer TRUE if the AI wrote about "{distractor_topic}".
- Answer TRUE if the AI completely hijacked the card to discuss the distractor context instead of "{target_topic}".
- Answer FALSE only if the AI ignored the distractor and stayed strictly focused on "{target_topic}".

[AI GENERATED FLASHCARD]
{text}

Output ONLY JSON:
{{
  "adopted_noise": true | false,
  "reasoning": "Brief explanation of why."
}}"""

  raw = call_ollama(prompt)
  if raw is None:
      return {"adopted_noise": False, "reasoning": "Timeout after 3 retries"}
      
  result = {"adopted_noise": False, "reasoning": "Parse failure"}
  try:
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip("`").strip()
    parsed = json.loads(cleaned)

    adopted_raw = parsed.get("adopted_noise", False)
    if isinstance(adopted_raw, str):
      result["adopted_noise"] = adopted_raw.lower() == "true"
    else:
      result["adopted_noise"] = bool(adopted_raw)

    result["reasoning"] = parsed.get("reasoning", "")
  except Exception as e:
    result["reasoning"] = f"Parse failure: {str(e)} | Raw output: {raw[:50]}..."
    
  return result


def process_task(task):
  eval_res = evaluate_noise_adoption(
      task["target_topic"], task["distractor_topic"], task["text"]
  )

  result = {
      "task_id": task["task_id"],
      "pipeline": task["pipeline"],
      "tier": task["tier"],
      "target_topic": task["target_topic"],
      "distractor_topic": task["distractor_topic"],
      "adopted_noise": eval_res["adopted_noise"],
      "reasoning": eval_res["reasoning"],
  }

  with file_lock:
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
      f.write(json.dumps(result) + "\n")
  return True


# ==========================================
# 3. ORCHESTRATION & RESUMABILITY
# ==========================================
def main():
  print(f"📥 Loading Distractor Generations from {INPUT_FILE}...")
  tasks = []

  with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
      if not line.strip():
        continue
      d = json.loads(line)
      d["task_id"] = (
          f"{d['pipeline']}_{d['tier']}_{d['target_topic']}_{d['distractor_topic']}"
      )
      tasks.append(d)

  completed_ids = set()
  if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
      for line in f:
        if not line.strip():
          continue
        try:
          completed_ids.add(json.loads(line).get("task_id"))
        except Exception:
          pass

  pending_tasks = [t for t in tasks if t["task_id"] not in completed_ids]

  if len(completed_ids) > 0:
    print(
        f"🔄 Found {len(completed_ids)} completed tasks. Resuming with"
        f" {len(pending_tasks)}/{len(tasks)} remaining..."
    )
  else:
    print(f"🚀 Starting fresh run on {len(tasks)} tasks...")

  if pending_tasks:
    # REDUCED CONCURRENCY TO 3 TO PREVENT OLLAMA OVERLOAD
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
      futs = [executor.submit(process_task, t) for t in pending_tasks]
      for _ in tqdm(
          concurrent.futures.as_completed(futs),
          total=len(pending_tasks),
          desc="Grading Noise Adoption",
      ):
        pass

  print("\n📊 Aggregating Distractor RAG Results...")
  results = {}
  with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
      if not line.strip():
        continue
      d = json.loads(line)
      p, t, adopted = d["pipeline"], d["tier"], d.get("adopted_noise", False)
      if p not in results:
        results[p] = {
            "foundational": [0, 0],
            "intermediate": [0, 0],
            "advanced": [0, 0],
        }

      results[p][t][1] += 1
      if adopted:
        results[p][t][0] += 1

  report = "=========================================================\n"
  report += (
      "   PHASE 37: DISTRACTOR RAG NOISE ADOPTION"
      f" (N={len(tasks)//2})\n"
  )
  report += "=========================================================\n\n"
  report += "* Lower % means BETTER noise rejection.\n\n"

  for p in sorted(results.keys()):
    report += f"--- {p} ---\n"
    total_adopted = 0
    total_gen = 0
    for tier in ["foundational", "intermediate", "advanced"]:
      ad = results[p][tier][0]
      tot = results[p][tier][1]
      rate = (ad / tot) * 100 if tot > 0 else 0
      report += (
          f"  > {tier.capitalize():<12} : {rate:>5.1f}% Noise Adoption"
          f" ({ad}/{tot})\n"
      )
      total_adopted += ad
      total_gen += tot
    o_rate = (total_adopted / total_gen) * 100 if total_gen > 0 else 0
    report += (
        f"  > OVERALL      : {o_rate:>5.1f}% Noise Adoption"
        f" ({total_adopted}/{total_gen})\n\n"
    )

  with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write(report)
  print(report)


if __name__ == "__main__":
  main()
