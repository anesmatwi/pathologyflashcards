import os
import json
import pandas as pd
import numpy as np
import scipy.stats as st
from sklearn.feature_extraction.text import TfidfVectorizer
import re

INPUT_FILE = "phase37_4way_multiseed_results.jsonl"
REPORT_FILE = "phase37_tfidf_lexical_separation.txt"

PIPELINES = {
    "p1_baseline_closed": "Baseline (Closed-Book)",
    "p2_baseline_open": "Baseline (Open-Book Optimal)",
    "p3_sft_closed": "Phase 37 SFT (Closed-Book)",
    "p4_sft_open": "Phase 37 SFT (Open-Book Optimal)"
}

def calc_stats(data_list):
    arr = np.array(data_list)
    if len(arr) < 2: return np.mean(arr), 0.0, 0.0
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    ci_margin = st.t.ppf(0.975, len(arr)-1) * st.sem(arr)
    return mean, std, ci_margin

def format_line(name, data):
    m, sd, ci = calc_stats(data)
    return f"{name:<35} : {m:>5.2f}% ± {ci:>4.2f}%  (SD: {sd:>4.2f}%)"

def extract_answer(text):
    try:
        match = re.search(r'\[ANSWER\](.*)(?:\[/ANSWER\]|$)', text, re.DOTALL | re.IGNORECASE)
        if match: return match.group(1).strip()
    except: pass
    return text.strip()

def get_top_terms(text, top_k=25):
    """Extracts the top 25 highest-weighted TF-IDF unigrams and bigrams."""
    if not text.strip(): return set()
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    try:
        tfidf_matrix = vectorizer.fit_transform([text])
        scores = zip(vectorizer.get_feature_names_out(), np.asarray(tfidf_matrix.sum(axis=0)).ravel())
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        return set([term for term, score in sorted_scores[:top_k]])
    except:
        return set()

def jaccard_similarity(set1, set2):
    """Calculates Jaccard overlap % between two term sets."""
    if not set1 and not set2: return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return (intersection / union) * 100 if union > 0 else 0.0

def main():
    print(f"📥 Loading generated matrix from {INPUT_FILE}...")
    
    results = {p: {"F_I": [], "I_A": [], "F_A": []} for p in PIPELINES}

    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            tiers_data = row.get("pipeline_outputs", {})
            
            for p_key in PIPELINES:
                text_f = extract_answer(tiers_data.get("foundational", {}).get(p_key, ""))
                text_i = extract_answer(tiers_data.get("intermediate", {}).get(p_key, ""))
                text_a = extract_answer(tiers_data.get("advanced", {}).get(p_key, ""))
                
                terms_f = get_top_terms(text_f)
                terms_i = get_top_terms(text_i)
                terms_a = get_top_terms(text_a)
                
                results[p_key]["F_I"].append(jaccard_similarity(terms_f, terms_i))
                results[p_key]["I_A"].append(jaccard_similarity(terms_i, terms_a))
                results[p_key]["F_A"].append(jaccard_similarity(terms_f, terms_a))

    report = "=========================================================\n"
    report += "   LEXICAL COGNITIVE SEPARATION (TF-IDF JACCARD OVERLAP) \n"
    report += "=========================================================\n"
    report += "* Lower % means BETTER separation (models are using distinct vocabulary per tier)\n"
    report += "* Higher % means POOR separation (models are just padding the same words)\n\n"

    comparisons = [
        ("1. Foundational vs Intermediate Overlap", "F_I"),
        ("2. Intermediate vs Advanced Overlap", "I_A"),
        ("3. Foundational vs Advanced Overlap (Total Shift)", "F_A")
    ]

    for title, key in comparisons:
        report += f"--- {title} ---\n"
        for p_key, p_name in PIPELINES.items():
            data = results[p_key][key]
            report += format_line(p_name, data) + "\n"
        report += "\n"

    print(report)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"✅ TF-IDF report saved to {REPORT_FILE}")

if __name__ == "__main__":
    main()
