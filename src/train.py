"""Configuration LoRA/QLoRA et entraînement du modèle.

QLoRA = le modèle de base est chargé en 4 bits (gelé), et seuls de petits adaptateurs LoRA
sont entraînés. C'est ce qui permet d'entraîner sur un GPU T4 gratuit (15 Go).
"""

import os
from pathlib import Path

BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

# Hyperparamètres par défaut (justifiés dans le notebook 03)
LORA_DEFAULTS = {
    "r": 16,                # rang des matrices A et B
    "lora_alpha": 32,       # facteur d'échelle (règle courante : 2 × r)
    "lora_dropout": 0.05,
    "target_modules": [     # toutes les couches linéaires du transformer
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
}

TRAINING_DEFAULTS = {
    "num_train_epochs": 2,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 4,   # batch effectif = 4 × 4 = 16
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 10,
    "max_length": 1024,
    "logging_steps": 10,
    "eval_strategy": "steps",
    "eval_steps": 50,
    "save_strategy": "steps",
    "save_steps": 50,
    "save_total_limit": 2,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "gradient_checkpointing": True,     # économise la mémoire GPU
    "fp16": True,                       # le T4 ne supporte pas bf16
    "optim": "paged_adamw_8bit",
    "report_to": "none",
    "seed": 42,
}


# ---------------------------------------------------------------------------
# Données
# ---------------------------------------------------------------------------

def to_prompt_completion(records):
    """Transforme les exemples en format prompt / completion pour TRL.

    prompt     = message système + message utilisateur (schéma + question)
    completion = la requête SQL attendue

    Avec ce format, la loss n'est calculée que sur la completion :
    le modèle apprend à produire le SQL, pas à recopier le schéma.
    """
    from datasets import Dataset

    rows = [{"prompt": r["messages"][:-1], "completion": [r["messages"][-1]]} for r in records]
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Modèle
# ---------------------------------------------------------------------------

def load_quantized_model(model_name=BASE_MODEL):
    """Charge le modèle de base en 4 bits (NF4) et son tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, dtype=torch.float16, device_map="auto"
    )
    model.config.use_cache = False  # incompatible avec le gradient checkpointing
    return model, tokenizer


def make_lora_config(**overrides):
    from peft import LoraConfig

    params = {**LORA_DEFAULTS, **overrides}
    return LoraConfig(bias="none", task_type="CAUSAL_LM", **params)


def make_training_args(output_dir, **overrides):
    from trl import SFTConfig

    params = {**TRAINING_DEFAULTS, **overrides}
    return SFTConfig(output_dir=output_dir, **params)


def build_trainer(model, tokenizer, train_ds, val_ds, lora_config, training_args):
    from trl import SFTTrainer

    return SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=lora_config,
    )


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def latest_checkpoint(output_dir):
    """Renvoie le dernier checkpoint sauvegardé (pour reprendre après une coupure Colab)."""
    ckpts = sorted(Path(output_dir).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    return str(ckpts[-1]) if ckpts else None


def dir_size_mb(path):
    total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    return round(total / 1e6, 1)