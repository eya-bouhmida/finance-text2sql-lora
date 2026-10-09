"""Chargement du modèle et génération des requêtes SQL.

Utilisé à la fois pour la baseline (modèle de base) et pour le modèle fine-tuné (base + adaptateur LoRA).
"""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def load_model(model_name=BASE_MODEL, adapter_path=None):
    """Charge le tokenizer et le modèle en float16 sur GPU.

    Si `adapter_path` est donné, l'adaptateur LoRA est ajouté au modèle de base.
    Le même chargement est utilisé pour la baseline et le modèle fine-tuné :
    la comparaison est donc équitable.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"  # nécessaire pour la génération par lots
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate(model, tokenizer, messages_list, batch_size=8, max_new_tokens=256):
    """Génère une réponse pour chaque conversation (décodage glouton, déterministe).

    Renvoie (liste des sorties brutes, temps moyen par question en secondes).
    """
    outputs = []
    start = time.time()
    for i in range(0, len(messages_list), batch_size):
        batch = messages_list[i:i + batch_size]
        prompts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in batch
        ]
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        new_tokens = gen[:, enc["input_ids"].shape[1]:]
        outputs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
        print(f"\r{min(i + batch_size, len(messages_list))}/{len(messages_list)}", end="")
    print()
    return outputs, (time.time() - start) / max(len(messages_list), 1)