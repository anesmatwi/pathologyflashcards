# Execution Command: torchrun --nproc_per_node=3 2_train_phase37_sft_ddp.py
import os
import torch
from datasets import load_dataset

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Unsloth MUST be imported BEFORE trl to prevent <EOS_TOKEN> monkey-patching bugs.
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from trl import SFTConfig, SFTTrainer

# ==========================================
# 1. HARDWARE & PIPELINE CONFIGURATION
# ==========================================
MODEL_NAME = "unsloth/Qwen2.5-14B-Instruct-bnb-4bit"
DATASET_PATH = "phase37_1_sft_dataset.jsonl" 
OUTPUT_DIR = "pathology_tutor_qwen_14b_phase37_sft"
MAX_SEQ_LENGTH = 2048  # Increased from 2048 to safely accommodate 5-chunk noisy RAG contexts
DTYPE = None
LOAD_IN_4BIT = True

# ==========================================
# 2. DDP GPU ASSIGNMENT
# ==========================================
local_rank = int(os.environ.get("LOCAL_RANK", 0))
device_map = {"": local_rank}

if local_rank == 0:
    print(f"🚀 Initializing Phase 37 DDP Training on {torch.cuda.device_count()} GPUs...")
    print("📊 Training on Noise-Injected, MedGemma-Augmented SFT Dataset")

print(f"📥 Loading Qwen2.5-14B Base Model on GPU {local_rank}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
    device_map=device_map,
)

# ==========================================
# 3. ADVANCED PEFT: Rank-Stabilized LoRA (rsLoRA)
# ==========================================
if local_rank == 0:
    print("🧠 Applying Rank-Stabilized LoRA (rsLoRA) Adapters...")

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=True,
    use_dora=False,
    loftq_config=None,
)

# ==========================================
# 4. DATASET FORMATTING (ChatML)
# ==========================================
if local_rank == 0:
    print("📚 Formatting Phase 37 ChatML Dataset...")

tokenizer = get_chat_template(
    tokenizer,
    chat_template="chatml",
    mapping={
        "role": "role",
        "content": "content",
        "user": "user",
        "assistant": "assistant",
    },
)

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [
        tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False
        )
        for convo in convos
    ]
    return {"text": texts}

dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

if local_rank == 0:
    print(f"✅ Successfully loaded {len(dataset)} training rows.")

# ==========================================
# 5. SFT TRAINER CONFIGURATION
# ==========================================
if local_rank == 0:
    print("🔥 Initializing Phase 37 SFT Trainer...")

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": False,
        },
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8, 
        warmup_steps=100,
        num_train_epochs=2,  
        learning_rate=1e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir=OUTPUT_DIR,
        save_strategy="steps",
        save_steps=300,
        save_total_limit=2,
        neftune_noise_alpha=5,
        ddp_find_unused_parameters=False,
        ddp_broadcast_buffers=False,
    ),
)

# ==========================================
# 6. EXECUTE & SAVE
# ==========================================
if local_rank == 0:
    print("🏃‍♂️ Starting Phase 37 Multi-GPU SFT...")

trainer_stats = trainer.train()

if local_rank == 0:
    print(f"💾 Saving Phase 37 SFT adapters to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ Phase 37 SFT Complete! The model is ready for evaluation.")
