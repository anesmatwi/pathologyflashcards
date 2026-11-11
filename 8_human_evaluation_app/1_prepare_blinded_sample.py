import json
import random
import re
from difflib import SequenceMatcher

INPUT_FILE = "phase37_4way_multiseed_results.jsonl"
READABLE_OUTPUT = "grouped_readable_set.jsonl"
APP_OUTPUT = "blinded_eval_set.jsonl"

PIPELINES = [
    "p1_baseline_closed", 
    "p2_baseline_open", 
    "p3_sft_closed", 
    "p4_sft_open"
]

def extract_qa(text):
    if not text: return None, None
    q_match = re.search(r'\[QUESTION\]\s*(.*?)\s*\[ANSWER\]', text, re.DOTALL | re.IGNORECASE)
    a_match = re.search(r'\[ANSWER\]\s*(.*?)(?:\[/ANSWER\]|$)', text, re.DOTALL | re.IGNORECASE)
    if q_match and a_match:
        return q_match.group(1).strip(), a_match.group(1).strip()
        
    try:
        q_part = re.search(r'(?:Question|Clinical Question)[\*:#\s]*(.*?)(?=(?:\n\s*)*[\*#]*\s*Answer[\*:#\s]*)', text, re.DOTALL | re.IGNORECASE)
        a_part = re.search(r'Answer[\*:#\s]*(.*)', text, re.DOTALL | re.IGNORECASE)
        if q_part and a_part:
            q = re.sub(r'[\*#]+$', '', q_part.group(1).strip()).strip()
            return q, a_part.group(1).strip()
    except Exception:
        pass
    return None, None

def is_semantic_duplicate(t1, t2):
    # Normalize strings for comparison
    a = t1.lower().replace(" ", "")
    b = t2.lower().replace(" ", "")
    
    # Catch acronyms and explicit overlaps
    if (a == "alp" and "alkalinephosphatase" in b) or (b == "alp" and "alkalinephosphatase" in a):
        return True
    
    # Check for direct substring nesting
    if len(a) > 4 and len(b) > 4:
        if a in b or b in a: return True
        
    # Check for heavy string similarity (e.g., "Bone Necrosis" vs "Necrosis of Bone")
    if SequenceMatcher(None, a, b).ratio() > 0.75:
        return True
        
    return False

def main():
    print(f"📥 Loading dataset from {INPUT_FILE}...")
    
    valid_topics_by_tier = {"foundational": {}, "intermediate": {}, "advanced": {}}
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            topic = data.get("seed_topic", "Unknown")
            
            for tier in ["foundational", "intermediate", "advanced"]:
                outputs = data.get("pipeline_outputs", {}).get(tier, {})
                topic_cards = {}
                for p in PIPELINES:
                    text = outputs.get(p, "")
                    q, a = extract_qa(text)
                    if q and a:
                        topic_cards[p] = {"topic": topic, "tier_true": tier, "model_true": p, "q": q, "a": a}
                
                if len(topic_cards) == 4:
                    valid_topics_by_tier[tier][topic] = topic_cards

    # Lock in your exact original 24 topics
    random.seed(42) 
    selected_topics_by_tier = {}
    for tier in ["foundational", "intermediate", "advanced"]:
        available_topics = list(valid_topics_by_tier[tier].keys())
        selected_topics_by_tier[tier] = random.sample(available_topics, min(8, len(available_topics)))

    # Post-processing: Apply semantic deduplication
    approved_topics = []
    
    for tier in ["foundational", "intermediate", "advanced"]:
        for i in range(len(selected_topics_by_tier[tier])):
            current_topic = selected_topics_by_tier[tier][i]
            
            # Check against LMP-1 and semantic overlap with previously approved topics
            is_duplicate = False
            if current_topic == "LMP-1":
                is_duplicate = True
            else:
                for approved in approved_topics:
                    if is_semantic_duplicate(current_topic, approved):
                        is_duplicate = True
                        break
            
            if is_duplicate:
                # Find a safe replacement
                available_replacements = [
                    t for t in valid_topics_by_tier[tier].keys() 
                    if t not in selected_topics_by_tier[tier] and t != "LMP-1"
                ]
                available_replacements.sort()
                
                safe_replacements = []
                for rep in available_replacements:
                    rep_conflict = False
                    for approved in approved_topics:
                        if is_semantic_duplicate(rep, approved):
                            rep_conflict = True
                            break
                    if not rep_conflict:
                        safe_replacements.append(rep)
                
                if safe_replacements:
                    replacement = random.sample(safe_replacements, 1)[0]
                    selected_topics_by_tier[tier][i] = replacement
                    approved_topics.append(replacement)
            else:
                approved_topics.append(current_topic)

    # Build the final JSONL files
    grouped_readable_set = []
    flat_app_set = []
    
    for tier in ["foundational", "intermediate", "advanced"]:
        for topic in selected_topics_by_tier[tier]:
            cards_dict = {}
            for p in PIPELINES:
                card = valid_topics_by_tier[tier][topic][p].copy()
                cards_dict[p] = card
                flat_app_set.append(card)
                
            grouped_readable_set.append({
                "topic": topic,
                "tier_true": tier,
                "pipeline_comparisons": cards_dict
            })

    # Save Readable Grouped JSONL
    grouped_readable_set.sort(key=lambda x: x["tier_true"])
    with open(READABLE_OUTPUT, "w", encoding="utf-8") as f:
        for group in grouped_readable_set:
            f.write(json.dumps(group) + "\n")
            
    # Save Flattened App JSONL
    random.seed(101)
    random.shuffle(flat_app_set)
    for i, card in enumerate(flat_app_set):
        card["eval_id"] = i + 1

    with open(APP_OUTPUT, "w", encoding="utf-8") as f:
        for card in flat_app_set:
            f.write(json.dumps(card) + "\n")
            
    print(f"✅ Created 24 Grouped Topic Blocks for human review -> {READABLE_OUTPUT}")
    print(f"✅ Created 96 Randomized Independent Cards for Streamlit -> {APP_OUTPUT}")

if __name__ == "__main__":
    main()
