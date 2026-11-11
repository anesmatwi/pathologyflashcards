# python compile_resident.py
# cp Modelfile_s pathology_tutor_qwen_14b_phase37_sft_gguf/Modelfile
# ollama create pathology_tutor_phase37_sft -f pathology_tutor_phase37_sft_gguf/Modelfile

import os
from unsloth import FastLanguageModel

CHECKPOINT_PATH = "/mnt/fanxiangssd/flashcardprojectv2phase9/pathology_tutor_qwen_14b_phase37_sft"
OUTPUT_DIR = "pathology_tutor_qwen_14b_phase37_sft"

print(f"📥 Loading Base Model and Adapters simultaneously from {CHECKPOINT_PATH}...")
# Because we repaired the folder, Unsloth can load it directly!
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = CHECKPOINT_PATH, 
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

print("\n💾 Merging weights and compiling Q4_K_M GGUF...")
# Now Unsloth recognizes its own PEFT structure and will merge correctly
model.save_pretrained_gguf(
    OUTPUT_DIR, 
    tokenizer, 
    quantization_method = "q4_k_m"
)

print(f"\n✅ Compilation complete! Output saved to: ./{OUTPUT_DIR}/")
