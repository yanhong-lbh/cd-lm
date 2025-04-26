import argparse
import torch
import os
import json
from collections import defaultdict
import re
from tqdm import tqdm
import numpy as np
import time

from trie import Trie
from utils import load_model_and_tokenizer


def save_data(file_path, data):
    with open(file_path, "w") as f:
        json.dump(data, f)


def load_data(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def read_token_probs(file_path):
    """Read a single token_probs JSON file, return None if it fails to decode."""
    try:
        with open(file_path, 'r') as f:
            token_probs = json.load(f)
        return token_probs
    except json.decoder.JSONDecodeError:
        print(f"Error decoding file: {file_path}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Create the token trie info from token probabilities.")
    parser.add_argument("--token_prob_dir", type=str, required=True,
                        help="Directory containing token_probs_*.json files.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the processed token-prob data and the trie info.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Hugging Face model name/path for loading the tokenizer.")
    parser.add_argument("--dataset_name", type=str, default="my_dataset",
                        help="Name of the dataset (used in some special-casing, e.g. 'PII').")
    parser.add_argument("--model_ds", type=str, default="gpt2",
                        help="Short label for the model from which token probs were derived (used in file naming).")
    parser.add_argument("--partition", type=str, default="train",
                        choices=["train", "validation", "test"],
                        help="Which data partition these token-prob files correspond to.")
    parser.add_argument("--tok_prob_threshold", type=float, default=0.3,
                        help="Filter threshold above which a token's probability is considered 'significant'.")
    parser.add_argument("--token_prob_prefix", type=str, default="token_probs",
                        help="File prefix to match when scanning `token_prob_dir` (default: 'token_probs').")
    # Optional argument if you want a separate 'model_gen' identifier:
    parser.add_argument("--model_gen", type=str, default=None,
                        help="Short label for the model used to generate hidden states (if different from model_ds).")

    args = parser.parse_args()

    # If model_gen isn't provided, default to model_ds
    if args.model_gen is None:
        args.model_gen = args.model_ds

    # ----------------------------------------------------------------------
    # 1. Build file paths for saving intermediate JSON files
    # ----------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.join(args.output_dir, f"{args.model_ds}_{args.dataset_name}_{args.partition}")
    file_paths = {
        "token_prob_list": f"{base_name}_token_prob_list.json",
        "all_phrases": f"{base_name}_all_phrases_{args.tok_prob_threshold}.json",
        "all_filtered_tok_probs": f"{base_name}_all_filtered_tok_probs_{args.tok_prob_threshold}.json"
    }

    # ----------------------------------------------------------------------
    # 2. Load tokenizer
    # ----------------------------------------------------------------------
    tokenizer, _ = load_model_and_tokenizer(args.model_name, cache_dir=None)

    # ----------------------------------------------------------------------
    # 3. Gather token-prob files
    # ----------------------------------------------------------------------
    token_prob_list = []
    if os.path.exists(file_paths["token_prob_list"]):
        # If we have a pre-saved version, load it
        token_prob_list = load_data(file_paths["token_prob_list"])
    else:
        # Otherwise, read all token_probs_*.json in the directory
        filenames = [
            fn for fn in os.listdir(args.token_prob_dir)
            if fn.startswith(args.token_prob_prefix) and fn.endswith(".json")
        ]

        for filename in filenames:
            file_path = os.path.join(args.token_prob_dir, filename)
            token_probs = read_token_probs(file_path)
            if not token_probs:
                continue

            # Skip the first 64 tokens if dataset is not 'PII'
            if args.dataset_name == 'PII':
                token_prob_list.append(token_probs)
            else:
                token_prob_list.append(token_probs[64:])

        save_data(file_paths["token_prob_list"], token_prob_list)

    # ----------------------------------------------------------------------
    # 4. Create chunked phrases above threshold
    # ----------------------------------------------------------------------
    if os.path.exists(file_paths["all_phrases"]) and os.path.exists(file_paths["all_filtered_tok_probs"]):
        all_phrases = load_data(file_paths["all_phrases"])
        all_filtered_tok_probs = load_data(file_paths["all_filtered_tok_probs"])
    else:
        all_phrases = []
        all_filtered_tok_probs = []

        for toke_prob in tqdm(token_prob_list, desc="Constructing chunks"):
            phrases = []
            filtered_tok_probs = []

            current_phrase = []
            current_tok_probs = []
            starting_token_added = False
            start_idx = 0

            for i, (token, prob, _, _) in enumerate(toke_prob):
                if prob > args.tok_prob_threshold:
                    # If we just crossed threshold, add the preceding token as start if not done already
                    if (i > 0) and (not starting_token_added):
                        previous_token = toke_prob[i - 1][0]
                        starting_token_added = True
                        current_phrase.insert(0, previous_token)
                        current_tok_probs.insert(0, toke_prob[i - 1])

                    current_phrase.append(token)
                    current_tok_probs.append(toke_prob[i])
                else:
                    if current_phrase:
                        phrases.append(current_phrase)
                        current_phrase = []
                    if current_tok_probs:
                        filtered_tok_probs.append(current_tok_probs)
                        current_tok_probs = []
                    starting_token_added = False
                    start_idx = i

            # End of sequence, flush any leftover
            if current_phrase:
                phrases.append(current_phrase)
            if current_tok_probs:
                filtered_tok_probs.append(current_tok_probs)

            all_phrases.append(phrases)
            all_filtered_tok_probs.append(filtered_tok_probs)

        save_data(file_paths["all_phrases"], all_phrases)
        save_data(file_paths["all_filtered_tok_probs"], all_filtered_tok_probs)

    # ----------------------------------------------------------------------
    # 5. Further process to keep sub-chunks of length >= 2
    # ----------------------------------------------------------------------
    separated_phrases = []
    separated_filtered_tok_probs = []

    for phrases, filtered_tok_probs in zip(all_phrases, all_filtered_tok_probs):
        # For any chunk with more than 2 tokens, create sub-chunks by removing leading tokens
        extended_phrases = [
            phrase[i:]
            for phrase in phrases
            for i in range(len(phrase) - 1)
            if len(phrase) > 2
        ]
        extended_filtered_tok_probs = [
            tok_prob[i:]
            for tok_prob in filtered_tok_probs
            for i in range(len(tok_prob) - 1)
            if len(tok_prob) > 2
        ]

        # Keep original 2-token chunks
        short_phrases = [phrase for phrase in phrases if len(phrase) == 2]
        short_filtered_tok_probs = [tok_prob for tok_prob in filtered_tok_probs if len(tok_prob) == 2]

        separated_phrases.extend(extended_phrases)
        separated_phrases.extend(short_phrases)
        separated_filtered_tok_probs.extend(extended_filtered_tok_probs)
        separated_filtered_tok_probs.extend(short_filtered_tok_probs)

    # ----------------------------------------------------------------------
    # 6. Build and populate the Trie
    # ----------------------------------------------------------------------
    trie = Trie(
        token_trie_dir=args.output_dir,
        dataset=args.dataset_name,
        model_ds=args.model_ds,
        model_gen=args.model_gen,
        partition=args.partition,
        tok_prob_threshold=args.tok_prob_threshold
    )

    for i, (phrase_tokens, token_probs) in enumerate(zip(separated_phrases, separated_filtered_tok_probs)):
        trie.insert(phrase_tokens, i, token_probs)

    # Reassign IDs, merge node values, and set token IDs
    trie.reassign_ids()
    trie.reassign_values()
    trie.set_token_ids(tokenizer)

    # Save Trie data
    trie.save_token_trie_info()

    print(f"[INFO] Done creating token trie info. Intermediate files saved in {args.output_dir}")


if __name__ == '__main__':
    main()
