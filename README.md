# Pathology Flashcard Curriculum Generator (Qwen2.5-14B)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/Paper-Artificial%20Intelligence%20in%20Medicine-green)](#citation)
[![HuggingFace Models](https://img.shields.io/badge/Models-HuggingFace-yellow)](#models-and-data)

This repository contains the official code, data pipelines, and evaluation frameworks for the paper: **"Fine-Tuning a Compact 14B Large Language Model for Multi-Level, RAG-Resilient Pathology Flashcard Generation"**. 

Our framework distils a multi-agent teacher pipeline into a compact, locally deployable 14B parameter model (`Qwen2.5-14B-Instruct`). By using length-balanced Rank-Stabilized LoRA (rsLoRA) and noisy Retrieval-Augmented Generation (RAG) training, the fine-tuned model successfully generates well-structured medical flashcards across three distinct clinical difficulty levels (Foundational/MS1, Intermediate/Resident, Advanced/Fellow) while remaining highly resilient to irrelevant textbook noise.

---

## 🏗️ Pipeline & Repository Structure

The repository is structured sequentially across 9 steps to reproduce the exact pipeline from dataset curation to the final cross-evaluation audit.

```text
pathologyflashcards/
├── 1_curation/                              # 1. DATASET CURATION & RAG INDEXING
│   ├── 1_vision_pipeline.py                 # MiniCPM-V vision-language PDF OCR
│   ├── 2_flashcard_generation.py            # Multi-agent teacher synthesis (Command R 35B / MedGemma-27B)
│   ├── 3_vector_db_ingestion.py             # ChromaDB indexing & bge-large embeddings
│   └── train_data.jsonl                     # The distilled clinical seed vault
│
├── 2_build_train_model/                     # 2. FINE-TUNING & QUANTIZATION
│   ├── 1_build_phase37_dataset.py           # Length-balanced SFT assembly & noisy RAG formatting
│   ├── 2_train_phase37_sft_ddp.py           # Multi-GPU rsLoRA + NEFTune training via Unsloth/TRL
│   └── 3_compile_gguf.py                    # Quantizes merged adapters into 4-bit Q4_K_M GGUF
│
├── 3_generate_for_ablation/                 # 3. HOLD-OUT INFERENCE GENERATION
│   └── 1_generate_phase37_4way.py           # Generates 3,000 cards across 4 pipelines (Base/SFT, Open/Closed)
│
├── 4_ablation_scoring_pass1/                # 4. AUTOMATED PASS 1 (COMPARATIVE DIFFICULTY & LENGTH)
│   ├── 1_score_phase37_4way.py              # RISE-Judge-32B side-by-side Bloom's Taxonomy scoring
│   ├── 2_audit_lengths.py                   # Word-length target compliance auditing
│   ├── 3_report_phase37_bloom_sig.py        # Wilcoxon signed-rank significance for tier progression
│   └── 4_report_phase37_tfidf.py            # Lexical overlap separation via TF-IDF Jaccard
│
├── 5_poison_pill/                           # 5. ADVERSARIAL IN-DOMAIN NOISE
│   ├── 1_score_distractor_rag_resilience.py # Initial semantic distractor checks
│   └── 2_score_phase37_poison_pill.py       # MedGemma-27B semantic adoption check of false injected claims
│
├── 6_ablation_scoring_pass2/                # 6. AUTOMATED PASS 2 (ABSOLUTE DIFFICULTY)
│   ├── 1_score_full_dok_classification.py   # Isolated DOK/SOLO classification (Level 1, 2, 3)
│   └── 2_report_dok_significance_32b.py     # Significance testing for absolute difficulty boundaries
│
├── 7_distractor_testing/                    # 7. ADVERSARIAL OFF-TOPIC RAG & FORMATTING
│   ├── 1_generate_distractor_eval_100.py    # Generates N=300 cards with unrelated pathology textbook chunks
│   ├── 2_score_distractor_rag_100.py        # Evaluates context sycophancy/hijacking
│   └── 3_score_formatting_integrity.py      # Checks [QUESTION]/[ANSWER] syntax and conversational cleanliness
│
├── 8_human_evaluation_app/                  # 8. CLINICAL PILOT (N=96)
│   ├── 1_prepare_blinded_sample.py          # Random deduplicated selection of 24 paired topics
│   ├── 2_app.py                             # Streamlit web app for blinded multi-axial grading
│   └── 3_report_pathologist_validation.py   # Compiles Factuality, Safety, and Scope metrics
│
└── 9_compare_auto_vs_human/                 # 9. CROSS-EVALUATION ANALYSIS
    └── 1_compare_auto_vs_human_96.py        # Reconciles Pass 1 & Pass 2 limits vs. specialist human grading
🚀 Quick Start & InstallationPrerequisitesPython 3.10+NVIDIA GPU with $\ge$ 16GB VRAMCUDA Toolkit 12.1+Environment SetupBashgit clone [https://github.com/anesmatwi/pathologyflashcards.git](https://github.com/anesmatwi/pathologyflashcards.git)
cd pathologyflashcards
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
Running the Model Locally (Ollama)The fine-tuned model has been compiled into a 4-bit quantized GGUF format for highly efficient local inference.Install Ollama.Pull/load the model using the provided Modelfile:Bashcd 2_build_train_model
ollama create pathology-tutor -f Modelfile_s
ollama run pathology-tutor
Example Inference Prompt:For the best results, use the exact lightweight two-sentence role prompt used during training. No explicit length boundaries or structural tag instructions are required at inference time."You are a Foundational Pathology Tutor for Year-1 medical students. Generate a highly concise flashcard for the following topic: Gastrointestinal Stromal Tumor."📦 Models and DataUpon formal publication in Artificial Intelligence in Medicine, the following assets will be available on HuggingFace:Base Dataset: Clinical seed vault (train_data.jsonl)SFT Dataset: 12,000-card length-balanced training set (phase37_1_sft_dataset.jsonl)Evaluation Sets: The N=3,000 generation results (phase37_4way_multiseed_results.jsonl) and N=96 human pilot set (dr_honeine_evaluations.jsonl)Model Weights: Fine-tuned Qwen2.5-14B rsLoRA adapters and standalone 4-bit GGUF.(Links will be updated upon manuscript acceptance).📄 CitationIf you use this code or dataset in your research, please cite our paper:Code snippet@article{atwi2026pathologyflashcards,
  title={Fine-Tuning a Compact 14B Large Language Model for Multi-Level, RAG-Resilient Pathology Flashcard Generation},
  author={Atwi, Mahmoud and Honeine, Fouad and Atat, Rachad},
  journal={Artificial Intelligence in Medicine},
  year={2026},
  publisher={Elsevier}
}
📜 LicenseThis project is licensed under the MIT License - see the LICENSE file for details. Note that any underlying medical texts or base model weights retain their original respective licenses.
