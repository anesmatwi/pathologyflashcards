import streamlit as st
import json
import os

INPUT_FILE = "blinded_eval_set.jsonl"
OUTPUT_FILE = "dr_honeine_evaluations.jsonl"

st.set_page_config(page_title="Pathology Flashcard Evaluation", layout="wide")

@st.cache_data
def load_cards():
    if not os.path.exists(INPUT_FILE): return []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def load_evaluations():
    evals = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    evals[data["eval_id"]] = data
    return evals

def save_evaluation(evals_dict):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for eval_data in evals_dict.values():
            f.write(json.dumps(eval_data) + "\n")

cards = load_cards()
evaluations = load_evaluations()

if not cards:
    st.error(f"Could not find {INPUT_FILE}. Please run the preparation script.")
    st.stop()

if 'current_idx' not in st.session_state:
    st.session_state.current_idx = 0

with st.sidebar:
    st.header("Evaluation Rubric")
    st.markdown("""
    **1. Factual Correctness**
    Are the diagnostic criteria, markers, and treatments completely accurate according to WHO consensus?
    
    **2. Clinical Safety**
    Is the medical information Safe, Potentially Misleading, or entirely Unsafe?
    
    **3. Educational Usefulness**
    Is this a high-yield, accurate, and valuable card for a medical learner?
    
    **4. Educational Level (Target Audience)**
    Subjective assessment of the training level required to learn this material.
    """)
    
    with st.expander("📖 View Cognitive Task (Bloom's) Rubric", expanded=False):
        st.markdown("""
        **Grade based ONLY on the cognitive task and structural depth, NOT the rarity of the medical vocabulary.**
        
        **🟢 Tier 1: Foundational (Bloom’s 1-2 | Remember & Understand)**
        * **The Task:** Recall facts, define terms, or list basic features.
        * **Question Pattern:** "What is [Entity]?" or "What are the variants?" 
        * **Answer Structure:** Highly concise (1-2 sentences or a brief list).
        * **Evaluator Check:** If it asks for a simple definition or list—even of a rare disease—it is Tier 1.

        **🟡 Tier 2: Intermediate (Bloom’s 3-4 | Apply & Analyze)**
        * **The Task:** Analyze diagnostic criteria, compare/contrast distinct entities, or interpret staging/classification.
        * **Question Pattern:** "Key differences between X and Y" or "Diagnostic criteria for X."
        * **Answer Structure:** Moderately detailed. Explains how to tell two things apart or applies staging rules (e.g., IHC panels).
        * **Evaluator Check:** Differentiating between concepts without diving into exhaustive molecular mechanics = Tier 2.

        **🔴 Tier 3: Advanced (Bloom’s 5-6 | Evaluate & Synthesize)**
        * **The Task:** Synthesize complex systemic mechanisms, evaluate molecular/genetic pathways, or justify targeted therapies.
        * **Question Pattern:** "How do genetic alterations contribute?" or "What are the limitations of [System]?"
        * **Answer Structure:** Exhaustive and comprehensive. Multi-paragraph answers detailing molecular cascades or driver mutations.
        * **Evaluator Check:** Requires deep mechanistic explanation (the "how and why") = Tier 3.
        """)
        
    with st.expander("📖 View Cognitive Scope & Execution Rubric", expanded=False):
        st.markdown("""
        **Evaluate how well the answer scales to the specific cognitive demand of the question.** High scores go to cards that deliver exactly the required level of detail without over-explaining simple concepts or under-explaining complex ones.
        
        *(Note: Conversational framing or natural filler is perfectly acceptable and should not be penalized, provided the core medical explanation remains on-topic.)*
        
        * **3 - Perfectly Scoped (Excellent):** The depth and structure match the question exactly. Foundational definitions are tight and concise. Intermediate comparisons provide clear, moderately detailed differentiation. Advanced mechanistic questions are answered with deep, exhaustive explanations. The card provides exactly what the learner needs for that specific difficulty tier.
        * **2 - Minor Mismatch (Good but Inefficient):** The answer is medically accurate but slightly misses the ideal scope. For example, it might provide too much narrative background for a basic definition, or an advanced explanation might be slightly too brief to fully cover the required mechanisms. The core knowledge is there, but the length and depth are slightly misaligned with the task.
        * **1 - Severe Mismatch (Unusable / Textbook Dump):** The execution completely ignores the scope of the question. For example, answering a basic "What is X?" question with a meandering, multi-paragraph textbook chapter, or answering an advanced molecular pathway question with a single vague sentence.
        """)
        
    st.divider()
    
    # Calculate fully completed (must have all new metrics)
    fully_evaluated = sum(1 for e in evaluations.values() if "bloom_tier" in e and "scope" in e)
    st.success(f"**Total Progress: {fully_evaluated} / {len(cards)} Fully Completed**")
    
    st.subheader("Card Navigator")
    card_labels = []
    for c in cards:
        e_id = c['eval_id']
        if e_id in evaluations:
            if "bloom_tier" in evaluations[e_id] and "scope" in evaluations[e_id]:
                status = "✅ Evaluated"
            else:
                status = "⚠️ Needs Updates"
        else:
            status = "⏳ Pending"
        card_labels.append(f"Card {e_id} {status}")
        
    selected_label = st.selectbox("Jump to a specific card:", card_labels, index=st.session_state.current_idx)
    
    selected_idx = card_labels.index(selected_label)
    if selected_idx != st.session_state.current_idx:
        st.session_state.current_idx = selected_idx
        st.rerun()

st.title("Pathology Model Validation")

col_prev, col_prog, col_next = st.columns([1, 2, 1])
with col_prev:
    if st.button("⬅️ View Previous", disabled=st.session_state.current_idx == 0):
        st.session_state.current_idx -= 1
        st.rerun()
with col_prog:
    st.markdown(f"<h4 style='text-align: center;'>Reviewing Card {st.session_state.current_idx + 1} of {len(cards)}</h4>", unsafe_allow_html=True)
with col_next:
    if st.button("View Next ➡️", disabled=st.session_state.current_idx == len(cards) - 1):
        st.session_state.current_idx += 1
        st.rerun()

current_card = cards[st.session_state.current_idx]
existing_eval = evaluations.get(current_card["eval_id"], {})

st.divider()
st.subheader(f"Topic: {current_card['topic']}")

with st.container(border=True):
    st.markdown("**QUESTION:**")
    st.info(current_card['q'])
    st.markdown("**ANSWER:**")
    st.success(current_card['a'])

# Original options
opt_factual = ["5 - Completely Accurate", "4 - Minor Omission", "3 - Partially Accurate", "2 - Major Inaccuracy", "1 - Completely False"]
opt_safety = ["Safe", "Potentially Misleading", "Unsafe"]
opt_tier = ["Foundational (MS1 Level)", "Intermediate (Resident Level)", "Advanced (Fellow/Oncologist Level)", "Unsuitable / Unclear"]
opt_utility = ["5 - Excellent", "4 - Good", "3 - Average", "2 - Poor", "1 - Useless"]

# New Bloom's options
opt_bloom = ["Tier 1: Foundational (Bloom's 1-2)", "Tier 2: Intermediate (Bloom's 3-4)", "Tier 3: Advanced (Bloom's 5-6)", "Unclassifiable"]

# New Scope options
opt_scope = [
    "3 - Perfectly Scoped (Excellent)", 
    "2 - Minor Mismatch (Good but Inefficient)", 
    "1 - Severe Mismatch (Unusable / Textbook Dump)"
]

def get_idx(options_list, match_string):
    if not match_string: return None
    try: return options_list.index(match_string)
    except ValueError: return None

with st.form("evaluation_form"):
    st.markdown("### Assess this Generation")
    
    if existing_eval:
        if "bloom_tier" not in existing_eval or "scope" not in existing_eval:
            st.warning("⚠️ You have partially evaluated this card. Please complete the newly added metrics below.")
        else:
            st.success("✅ You have fully evaluated this card. You can modify your answers below.")
    
    col1, col2 = st.columns(2)
    with col1:
        factual = st.radio("Factual Correctness", opt_factual, index=get_idx(opt_factual, existing_eval.get("factual")))
        safety = st.radio("Clinical Safety", opt_safety, index=get_idx(opt_safety, existing_eval.get("safety")))
        utility = st.radio("Educational Usefulness", opt_utility, index=get_idx(opt_utility, existing_eval.get("utility")))
    
    with col2:
        tier_approp = st.radio("Educational Level (Target Audience)", opt_tier, index=get_idx(opt_tier, existing_eval.get("tier_assigned")))
        st.markdown("---")
        bloom_tier = st.radio("Cognitive Task (Bloom's Taxonomy) - *New*", opt_bloom, index=get_idx(opt_bloom, existing_eval.get("bloom_tier")))
        scope = st.radio("Cognitive Scope & Execution - *New*", opt_scope, index=get_idx(opt_scope, existing_eval.get("scope")))
        
    comments = st.text_area("Optional Comments / Notes on Errors", value=existing_eval.get("comments", ""))
    warning_placeholder = st.empty()
    submitted = st.form_submit_button("💾 Save & Advance", type="primary")
    
    if submitted:
        if None in [factual, safety, tier_approp, bloom_tier, utility, scope]:
            warning_placeholder.error("⚠️ Action Required: Please select an answer for all six radio button questions before saving.")
        else:
            evaluations[current_card["eval_id"]] = {
                "eval_id": current_card["eval_id"],
                "model_true": current_card["model_true"],
                "topic": current_card["topic"],
                "tier_true": current_card["tier_true"],
                "factual": factual,
                "safety": safety,
                "tier_assigned": tier_approp,
                "bloom_tier": bloom_tier,
                "utility": utility,
                "scope": scope,
                "comments": comments
            }
            save_evaluation(evaluations)
            
            next_idx = st.session_state.current_idx + 1
            if next_idx < len(cards):
                st.session_state.current_idx = next_idx
            st.rerun()
