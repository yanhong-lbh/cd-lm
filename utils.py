import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_model_and_tokenizer(model_path, cache_dir=None):
    """
    Load a model and tokenizer from a given Hugging Face model repository path.

    Args:
        model_path (str): The Hugging Face model identifier or local path.
        cache_dir (str, optional): Path to store or look for downloaded model files.
    
    Returns:
        (tokenizer, model): A tuple containing the tokenizer and model.
    """
    # Use use_fast=False to avoid certain edge cases with special tokens in some large models.
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir=cache_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device, dtype=torch.float32)
    model.eval()

    return tokenizer, model
