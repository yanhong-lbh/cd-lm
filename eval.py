import os
import time
import json
import torch
import argparse
from tqdm import tqdm

from lm_trie import LMWithTrie
from utils import load_model_and_tokenizer, save_json, load_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluation (Inference) script for chunk-distilled language modeling using a token-trie datastore."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path or identifier for the HuggingFace model (e.g. 'gpt2' or local path)."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        required=True,
        help="Directory for HF caching and for storing any intermediate trie files."
    )
    parser.add_argument(
        "--token_trie_dir",
        type=str,
        required=True,
        help="Directory containing the stored token trie subfolders (token ID h5/pkl files)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (used in naming conventions for output)."
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="test",
        help="Data partition for which the trie was built (train/validation/test)."
    )
    parser.add_argument(
        "--tok_prob_threshold",
        type=float,
        required=True,
        help="Token probability threshold used during trie creation (must match the one from make_token_trie.py)."
    )
    parser.add_argument(
        "--model_ds",
        type=str,
        required=True,
        help="Model name/identifier used for the datastore/trie creation."
    )
    parser.add_argument(
        "--model_gen",
        type=str,
        required=True,
        help="Model name/identifier used for generation (often same as model_ds, but can differ)."
    )
    parser.add_argument(
        "--no_reprocessing",
        action="store_true",
        help="If set, do NOT reprocess raw .h5 tries into .pkl. Use existing pickles if available."
    )
    parser.add_argument(
        "--move_trie_to_gpu",
        action="store_true",
        help="If set, moves all trie data to GPU memory up-front (can be large)."
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        required=True,
        help="JSON file containing the list of prompt strings to evaluate."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save generation results (.json)."
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.5,
        help="Cosine similarity threshold for retrieving next token chunk from the trie."
    )
    parser.add_argument(
        "--gen_baseline_only",
        action="store_true",
        help="If set, only do baseline LM generation (equivalent to setting similarity_threshold=1)."
    )
    parser.add_argument(
        "--exclude_huge_token_tries",
        action="store_true",
        help="If set, skip retrieval if the current token is in an 'excluded tries' list (particularly for large tries)."
    )
    parser.add_argument(
        "--excluded_tries_prior",
        type=str,
        default=None,
        help="JSON file containing a list of token IDs to be excluded from trie retrieval if --exclude_huge_token_tries is set."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="Number of tokens to generate beyond the prompt length."
    )
    return parser.parse_args()


def generate_results(
    input_text,
    lm_w_trie,
    tokenizer,
    max_length,
    gen_baseline_only,
    sim_threshold
):
    """
    Generate text from a given input_text, optionally retrieving from a token trie based on sim_threshold.
    If gen_baseline_only is True, does naive generation (threshold=1 => no retrieval).
    Returns (dict_of_generations, dict_of_timing).
    """
    results = {}
    times = {}

    # Convert input prompt to tokens
    input_tokens = tokenizer.encode(input_text)

    # Always store the prefix for reference
    results['prefix'] = input_text

    # If only baseline generation is requested
    if gen_baseline_only:
        start_time = time.time()
        # The 'generate()' method in LMWithTrie with similarity_threshold=1 => pure LM generation.
        generated_baseline = lm_w_trie.generate(
            input_tokens, max_length=max_length, similarity_threshold=1
        )
        end_time = time.time()
        times['baseline_generation_time'] = end_time - start_time

        # generated_baseline can be a tuple if retrieval is also returned; handle both forms
        if isinstance(generated_baseline, tuple):
            baseline_tokens = generated_baseline[0]
        else:
            baseline_tokens = generated_baseline

        results['baseline_generation_tokens'] = baseline_tokens.tolist()
        results['baseline_generation_text'] = tokenizer.decode(baseline_tokens[0])
        return results, times

    # Otherwise do normal retrieval-based generation
    start_time = time.time()
    generated, retrieved_phrases = lm_w_trie.generate_parallel(
        input_tokens, max_length=max_length, similarity_threshold=sim_threshold
    )
    end_time = time.time()

    times[f'generation_t{sim_threshold}'] = end_time - start_time

    results[f'generation_t{sim_threshold}_tokens'] = generated[0].tolist()
    results[f'generation_t{sim_threshold}_text'] = tokenizer.decode(generated[0])
    results[f'generation_t{sim_threshold}_retrieved_phrases'] = retrieved_phrases

    return results, times


def main():
    args = parse_args()

    # Load model & tokenizer
    tokenizer, model = load_model_and_tokenizer(
        model_path=args.model_path,
        cache_dir=args.cache_dir
    )
    model.eval()

    # Decide on device (if multiple GPUs, adjust as desired)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load excluded tries if needed
    excluded_tries = []
    if args.exclude_huge_token_tries and args.excluded_tries_prior is not None:
        if os.path.exists(args.excluded_tries_prior):
            with open(args.excluded_tries_prior, 'r') as f:
                excluded_tries = json.load(f)

    # Instantiate the LMWithTrie wrapper
    lm_w_trie = LMWithTrie(
        lm=model,
        tokenizer=tokenizer,
        token_trie_dir=args.token_trie_dir,
        dataset=args.dataset,
        partition=args.partition,
        tok_prob_threshold=args.tok_prob_threshold,
        model_ds=args.model_ds,
        model_gen=args.model_gen,
        cache_dir=args.cache_dir,
        no_reprocessing=args.no_reprocessing,
        move_trie_to_gpu=args.move_trie_to_gpu,
        exclude_huge_token_tries=args.exclude_huge_token_tries,
        excluded_tries_prior=excluded_tries
    )

    # Prepare output directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Load prompts from JSON
    prompts = load_json(args.prompt_file)
    if not isinstance(prompts, list):
        raise ValueError("Your prompt_file must contain a list of prompt strings.")

    for i, prompt in enumerate(prompts):
        # (Optional) Format prompt like original code with special tokens. Adjust as needed.
        input_text = f"<|USER|> {prompt} <|ASSISTANT|> "
        prefix_len = len(tokenizer.encode(input_text))
        max_length = prefix_len + args.max_new_tokens

        # Each prompt result is saved as an individual file
        save_path = os.path.join(args.save_dir, f"prompt_{i}.json")
        if os.path.exists(save_path):
            print(f"Skipping prompt index {i} because file already exists: {save_path}")
            continue

        # Generate
        results, timing = generate_results(
            input_text=input_text,
            lm_w_trie=lm_w_trie,
            tokenizer=tokenizer,
            max_length=max_length,
            gen_baseline_only=args.gen_baseline_only,
            sim_threshold=args.similarity_threshold
        )

        # Consolidate info
        output_to_save = {
            "generation": results,
            "time": timing
        }

        # Save
        save_json(output_to_save, save_path)
        print(f"Saved generation output for prompt {i} at: {save_path}")


if __name__ == "__main__":
    main()
