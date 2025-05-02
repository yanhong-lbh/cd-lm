#!/usr/bin/env python

import os
import json
import torch
import numpy as np
import argparse
import math
import re
from tqdm import tqdm
from pathlib import Path
import h5py
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoConfig


def load_texts(json_file):
    """
    Loads a JSON file with entries of the form:
    [{'text': '...'}]
    Returns a list of strings.
    """
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [entry["text"] for entry in data]

def tokenize_passages(passages, tokenizer):
    """
    Convert each passage from text to a tensor of token IDs.
    """
    tokenized_passages = []
    for passage in passages:
        token_ids = tokenizer.encode(passage, add_special_tokens=False)
        tokenized_passages.append(torch.tensor(token_ids, dtype=torch.long))
    return tokenized_passages

def chunk_passages(tokenized_passages, chunk_size=512, stride=448, min_length=64):
    """
    Given a list of tokenized passages, create overlapping chunks:
    - chunk_size: total length of each chunk
    - stride: how many tokens to move between consecutive chunks
    - min_length: ignore passages shorter than this
    """
    chunked_passages = []
    for passage in tokenized_passages:
        if len(passage) < min_length:
            continue
        num_chunks = (len(passage) - (chunk_size - stride)) // stride + 1
        for i in range(num_chunks):
            start = i * stride
            end = start + chunk_size
            chunk = passage[start:end]
            chunked_passages.append(chunk)
    return chunked_passages

def calculate_token_probabilities(tokens, model, tokenizer, chunk_id, save_hidden_states=False):
    """
    Given a chunk of token ids, compute token-level probabilities
    and optionally hidden states from the last layer.

    Returns:
        token_prob_pairs: list of (token_str, probability, chunk_id, position)
        hidden_states (if requested)
    """
    device = next(model.parameters()).device

    # Reshape tokens to [batch_size=1, seq_length]
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)

    tokens = tokens.to(device)

    with torch.no_grad():
        outputs = model(tokens, labels=tokens, output_hidden_states=save_hidden_states)
    
    logits = outputs.logits  # [batch_size, seq_len, vocab_size]
    # Shift left to align with next token predictions
    logits = logits[:, :-1, :]
    labels = tokens[:, 1:]

    probs = torch.softmax(logits, dim=-1)
    # Probability of the *actual* label at each time-step
    gathered_probs = probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    gathered_probs = gathered_probs.squeeze(0).cpu()
    tokens_cpu = tokens.squeeze(0).cpu()

    # Build readable token strings
    token_strings = [tokenizer.decode([tid], clean_up_tokenization_spaces=False) 
                     for tid in tokens_cpu.tolist()]

    # Pad first token's probability with 0.0, or some placeholder
    token_prob_pairs = [(token_strings[0], 0.0, chunk_id, 0)]
    for idx in range(1, len(token_strings)):
        token_prob_pairs.append(
            (token_strings[idx], float(gathered_probs[idx-1].item()), chunk_id, idx)
        )

    if save_hidden_states:
        # For HF models, hidden_states is a tuple of all layers,
        # final_hidden_states is outputs.hidden_states[-1] with shape [1, seq_len, hidden_dim]
        final_hidden_states = outputs.hidden_states[-1].squeeze(0).cpu().numpy()
        return token_prob_pairs, final_hidden_states

    return token_prob_pairs, None


def get_model_dim(model_path):
    """
    Tries to infer the hidden/embedding dimension from the config of the
    Hugging Face model at model_path.

    We try several attribute names commonly used by different model configs.
    """
    config = AutoConfig.from_pretrained(model_path)
    possible_attrs = ["hidden_size", "n_embd", "d_model"]
    for attr in possible_attrs:
        if hasattr(config, attr):
            return getattr(config, attr)

    raise ValueError(f"Could not determine hidden state dimension from config at {model_path}.")


def convert_npy_to_dat(file_dir, start_idx, end_idx, model_path, offset=0):
    """
    Convert .npy hidden states into .dat memmap. 
    offset can be used if you have special indexing (like skipping).
    """
    for k in range(start_idx, end_idx):
        npy_file = file_dir / f'hidden_states_{k+offset}.npy'
        if not npy_file.exists():
            # You could handle 'missing file' if relevant in your pipeline
            continue

        hs_npy = np.load(npy_file)
        # For LLaMA-based or Mistral-based models, the shape might need a squeeze
        if 'llama' in model_path.lower() or 'mistral'in model_path.lower()::
            hs_npy = hs_npy.squeeze()
        hs_dat_file = file_dir / 'temp_dat' / f'{k}_l{hs_npy.shape[0]}.dat'

        if not hs_dat_file.exists():
            fp = np.memmap(hs_dat_file, dtype='float32', mode='w+', shape=hs_npy.shape)
            fp[:] = hs_npy[:]

def construct_memmap(args):
    """
    Build a large memmapped matrix from the hidden states that were produced
    by the extraction step.

    Steps:
    1) Convert .npy to .dat using memmap chunk by chunk.
    2) Gather per-chunk sequence lengths in `index_to_seqlen`.
    3) Concatenate .dat files in sets to form big .dat, then combine again.
    4) Store final matrix in .h5 as well.

    This is a direct combination of logic from construct_memmapped_matrix.py
    adapted to argument-based usage.
    """
    # 1. Prepare directories
    file_dir = Path(args.output_dir) / args.partition
    hs_dat_path = file_dir / 'temp_dat'
    hs_dat_path.mkdir(exist_ok=True)

    # 2. Gather all .npy files
    all_npy_files = sorted(list(file_dir.glob('hidden_states_*.npy')), 
                           key=lambda x: int(x.stem.split('_')[2]))
    total_file_len_npy = len(all_npy_files)

    # 3. Convert all .npy -> .dat
    # This is multi-threaded. You can adjust max_workers.
    with ThreadPoolExecutor(max_workers=16) as executor:
        # We pass range(0, total_file_len_npy) to convert
        executor.map(lambda k: convert_npy_to_dat(file_dir, k, k+1, args.model_path, offset=0),
                     range(total_file_len_npy))

    # 4. Verify the number of .dat matches .npy
    all_dat_files = list(hs_dat_path.glob('*.dat'))
    total_file_len_dat = len(all_dat_files)
    if total_file_len_npy != total_file_len_dat:
        raise ValueError("The number of .npy files doesn't match the number of .dat files. "
                         "Check for missing or partial conversions.")

    # 5. Build index_to_seqlen
    index_to_seqlen = {}
    for file in all_dat_files:
        # e.g. "53_l512.dat" -> index=53, seq_len=512
        # splitted = file.name.split('_') => ["53", "l512.dat"]
        index_str = file.name.split('_')[0]
        seq_part = file.name.split('_')[1]  # "l512.dat"
        seq_len = int(re.findall(r'\d+', seq_part)[0])  # 512
        index_to_seqlen[int(index_str)] = seq_len

    # 6. Save index_to_seqlen as JSON
    save_dir = Path(args.output_dir)
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / f'{args.partition}_index_to_seqlen.json'
    with save_path.open('w') as f:
        json.dump(index_to_seqlen, f)

    # 7. Combine all .dat in chunks of 500
    total_file_len = total_file_len_npy
    model_dim = get_model_dim(args.model_path)
    n_of_k = math.ceil(total_file_len / 500)

    for k in range(n_of_k):
        filename = save_dir / f'{args.partition}_{k}.dat'
        if filename.exists():
            continue

        start_idx = k * 500
        end_idx = min(total_file_len, (k + 1) * 500)
        total_seq_len = 0
        for i in range(start_idx, end_idx):
            total_seq_len += index_to_seqlen[i]

        all_hidden_states = np.memmap(filename, dtype='float32', mode='w+', shape=(total_seq_len, model_dim))
        pos = 0
        for i in tqdm(range(start_idx, end_idx), desc=f"Combining chunk sets {k}/{n_of_k}"):
            seq_len = index_to_seqlen[i]
            chunk_file = hs_dat_path / f"{i}_l{seq_len}.dat"
            chunk_hs = np.memmap(chunk_file, dtype='float32', mode='r', shape=(seq_len, model_dim))
            all_hidden_states[pos : pos + seq_len] = chunk_hs[:]
            pos += seq_len

    # 8. Summaries for chunk sets
    set_to_seqlen = {}
    for k in range(n_of_k):
        start_idx = k * 500
        end_idx = min(total_file_len, (k + 1) * 500)
        total_seq_len = 0
        for i in range(start_idx, end_idx):
            total_seq_len += index_to_seqlen[i]
        set_to_seqlen[k] = total_seq_len

    save_path = save_dir / f'{args.partition}_set_to_seqlen.json'
    with save_path.open('w') as f:
        json.dump(set_to_seqlen, f)

    # 9. Combine chunk sets into final .dat
    filename = save_dir / f'{args.partition}.dat'
    total_seq_len = sum(set_to_seqlen.values())

    all_hidden_states = np.memmap(filename, dtype='float32', mode='w+', shape=(total_seq_len, model_dim))
    pos = 0
    for k in tqdm(range(n_of_k), desc="Combining all sets into final .dat"):
        seq_len = set_to_seqlen[k]
        file_path = save_dir / f"{args.partition}_{k}.dat"
        chunk_hs = np.memmap(file_path, dtype='float32', mode='r', shape=(seq_len, model_dim))
        all_hidden_states[pos : pos + seq_len] = chunk_hs[:]
        pos += seq_len

    # 10. Build index_to_acc_index for retrieval
    pos = 0
    index_to_acc_index = {}
    for i in range(total_file_len):
        index_to_acc_index[i] = pos
        seq_len = index_to_seqlen[i]
        pos += seq_len

    save_path = save_dir / f'{args.partition}_index_to_acc_index.json'
    with save_path.open('w') as f:
        json.dump(index_to_acc_index, f)

    # 11. Also store final matrix in h5
    h5_file_path = save_dir / f'{args.partition}.h5'
    with h5py.File(h5_file_path, 'w') as h5_file:
        h5_file.create_dataset('all_hidden_states', data=all_hidden_states)

    print(f"[Done] Memmap and h5 built at: {filename} and {h5_file_path}.")

def main():
    parser = argparse.ArgumentParser()
    # Arguments needed from original extract_chunks.py
    parser.add_argument("--model_path", type=str, required=True,
                        help="Hugging Face model identifier (or local path).")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory containing train.json, validation.json, and test.json.")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save token probabilities and hidden states.")
    parser.add_argument("--save_hidden_states", action="store_true",
                        help="Whether to compute and save hidden states for each chunk.")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Directory for caching model weights.")
    parser.add_argument("--partition", type=str, default="train",
                        help="Which split to process: train/validation/test")
    parser.add_argument("--construct_memmap", type=bool, default=False,
                        help="If True, run the memmap construction step after extraction.")
    parser.add_argument("--chunk_size", type=int, default=512,
                        help="Size of each chunk in tokens.")
    parser.add_argument("--stride", type=int, default=448,
                        help="Stride for overlapping chunks.")
    parser.add_argument("--min_length", type=int, default=64,
                        help="Minimum passage length (tokens) to consider.")
    args = parser.parse_args()


    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Loading tokenizer/model from {args.model_path} (cache_dir={args.cache_dir}) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, cache_dir=args.cache_dir)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    splits = [args.partition] 

    for split in splits:
        split_file = os.path.join(args.data_dir, f"{split}.json")
        if not os.path.exists(split_file):
            print(f"WARNING: {split_file} not found. Skipping {split} split.")
            continue

        passages = load_texts(split_file)
        print(f"Loaded {len(passages)} passages for {split} split.")

        # 2.2 Tokenize
        print(f"Tokenizing {split} passages...")
        tokenized_passages = tokenize_passages(passages, tokenizer)

        # 2.3 Chunk them
        print(f"Chunking {split} passages (chunk_size={args.chunk_size}, stride={args.stride})...")
        chunked_passages = chunk_passages(
            tokenized_passages,
            chunk_size=args.chunk_size,
            stride=args.stride,
            min_length=args.min_length
        )
        print(f"Total chunks for {split}: {len(chunked_passages)}")

        # 2.4 For each chunk, compute probabilities (& hidden states if requested)
        split_output_dir = os.path.join(args.output_dir, split)
        os.makedirs(split_output_dir, exist_ok=True)

        for index, chunk in tqdm(enumerate(chunked_passages), desc=f"Processing {split} chunks"):
            token_probs_json_path = os.path.join(split_output_dir, f"token_probs_{index}.json")
            hidden_states_path = os.path.join(split_output_dir, f"hidden_states_{index}.npy")

            # Avoid re-computing if files already exist
            if os.path.exists(token_probs_json_path) and \
               (not args.save_hidden_states or os.path.exists(hidden_states_path)):
                continue

            token_prob_pairs, hidden_states = calculate_token_probabilities(
                chunk, model, tokenizer, index, save_hidden_states=args.save_hidden_states
            )

            # 2.4.1 Save token probabilities
            if not os.path.exists(token_probs_json_path):
                with open(token_probs_json_path, "w", encoding="utf-8") as f:
                    json.dump(token_prob_pairs, f, ensure_ascii=False, indent=2)

            # 2.4.2 Save hidden states if requested
            if args.save_hidden_states and hidden_states is not None and not os.path.exists(hidden_states_path):
                np.save(hidden_states_path, hidden_states)

        print(f"Finished {split} split. Results saved in {split_output_dir}.")

    # 3. If requested, construct memmap from the new hidden states.
    if args.construct_memmap and args.save_hidden_states:
        print("\n[Step] Constructing memmap from saved hidden states...")
        construct_memmap(args)
    elif args.construct_memmap and not args.save_hidden_states:
        print("\n[Warning] --construct_memmap was set, but --save_hidden_states was not set. "
              "You need hidden states to build the memmap. Skipping memmap step.")

if __name__ == "__main__":
    main()
