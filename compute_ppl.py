#!/usr/bin/env python

import argparse
import os
import math
import json
import torch
import torch.nn.functional as F
import glob
import numpy as np
import re
from concurrent.futures import ProcessPoolExecutor

from utils import load_json, save_json, load_model_and_tokenizer
from lm_trie import LMWithTrie


def save_values(dir_path, file_name, data):
    """
    Saves data (tensor, list, etc.) into dir_path/file_name via torch.save
    """
    os.makedirs(dir_path, exist_ok=True)
    torch.save(data, os.path.join(dir_path, file_name))


def compute_ppl_lm_baseline_only(k, lm_pred_probs_dir, lm_ppl_dir):
    """
    Compute PPL for the LM baseline only for example k.
    """
    with open(f"{lm_pred_probs_dir}/{k}.json", "r") as f:
        lm_pred_probs = json.load(f)

    total_ce_lm_only = -1 * sum([math.log(prob) for prob in lm_pred_probs])
    ppl = math.exp(total_ce_lm_only / len(lm_pred_probs))

    with open(f"{lm_ppl_dir}/{k}.json", "w") as f:
        json.dump(ppl, f)


def compute_seq_ppl(
    k,
    base_dir,
    trie_ppl_dir,
    lm_pred_probs_dir,
    ground_truth_dir,
    retrieved_dir,
    dataset,
    partition,
    q_func_name,
    q_func_max_prob,
    tok_prob_threshold,
    similarity_threshold
):
    """
    Computes the sequence perplexity for a single example k.
    """

    # If the output for this example already exists, skip
    if os.path.exists(f"{trie_ppl_dir}/{k}.json"):
        return

    # Load the ground truth (drop first token with [1:])
    ground_truth = torch.load(f"{ground_truth_dir}/{k}.pt")[1:]

    # Load the LM-predicted probabilities
    with open(f"{lm_pred_probs_dir}/{k}.json", "r") as f:
        lm_pred_probs = json.load(f)

    retrieved_path = f"{retrieved_dir}/{k}.json"

    with open(retrieved_path, "r") as f:
        retrieved_sim = json.load(f)

    seq_len = len(ground_truth)

    lengths = {}
    similarities = {}
    ret_ids = {}

    # Prepare retrieved chunk info
    for key in retrieved_sim.keys():
        if retrieved_sim[key] is not None:
            ret_ids[int(key)] = retrieved_sim[key][0]
            lengths[int(key)] = len(ret_ids[int(key)])
            similarities[int(key)] = retrieved_sim[key][1]

    # Define q_func exactly as in your original code
    def q_func(s):
        if q_func_name == "identity":
            return s if s < 1 else (1 - 1e-12)

        elif q_func_name.startswith("map"):
            def str_to_float(ss):
                for i, char in enumerate(ss):
                    if char != '0':
                        return float(ss[:i] + '.' + ss[i:])
                return 0.0

            # Extract A, B from the q_func_name
            raw_A_and_B = re.compile(r'\d+').findall(q_func_name)[:2]
            A, B = [str_to_float(x) for x in raw_A_and_B]

            similarity = s
            if similarity == A:
                s = B
            elif similarity < A:
                s = (B / A) * similarity
                if q_func_name.endswith("_0"):
                    s = 0
            else:
                max_prob = 1
                if q_func_max_prob > 0:
                    max_prob = q_func_max_prob
                s = B + ((max_prob - B) / (max_prob - A)) * (similarity - A)
                s = s if s < max_prob else (max_prob - 1e-12)
            return s

    # Truncate the retrieved chunks if they exceed the sequence length
    offset = {i: i + len(v) - (seq_len - 1) for i, v in ret_ids.items()}
    ret_ids = {i: v[:-offset[i]] if offset[i] > 0 else v for i, v in ret_ids.items()}
    ret_ids = {i: v for i, v in ret_ids.items() if len(v) > 0}

    # Increase by one for storing prob at first token
    seq_len = len(ground_truth) + 1

    # Filter out retrieved chunks that contain at least one wrong token
    valid_ret_ids = {
        i+1: v if list(ground_truth[i: i+len(v)]) == v else None
        for i, v in ret_ids.items()
    }

    # Shift similarities by +1 to match valid_ret_ids
    similarities = {i+1: v for i, v in similarities.items()}

    # Create two lists for seq length, initialized with all 0 or None
    z_1_prob = [0 if i in valid_ret_ids.keys() else None for i in range(seq_len)]
    z_0_prob = [0] * seq_len

    reversed_list = list(range(seq_len))
    reversed_list.reverse()

    for i in reversed_list:
        if i == seq_len - 1:  # last token in the seq
            if z_1_prob[i] is not None:
                q = q_func(similarities[i])
                log_q = 0 if q == 0 else math.log(q)
                valid_ret = valid_ret_ids[i]
                if valid_ret is not None:
                    z_1_prob[i] = log_q * 1
                else:
                    z_1_prob[i] = q * 0
            else:
                q = 0
                z_1_prob[i] = 0

            z_0_prob[i] = math.log((1 - q) * lm_pred_probs[i-1])
        else:
            if z_1_prob[i] is not None:
                q = q_func(similarities[i])
                log_q = 0 if q == 0 else math.log(q)
                valid_ret = valid_ret_ids[i]
                if valid_ret is not None:
                    next_idx = i + len(valid_ret)
                    if z_1_prob[next_idx] == 0:
                        z_1_prob[i] = log_q + z_0_prob[next_idx]
                    else:
                        z_1_prob[i] = log_q + np.logaddexp(z_0_prob[next_idx], z_1_prob[next_idx])
                else:
                    z_1_prob[i] = q * 0
            else:
                q = 0
                z_1_prob[i] = 0

            if z_1_prob[i+1] == 0:
                z_0_prob[i] = math.log(1 - q) + math.log(lm_pred_probs[i-1]) + z_0_prob[i+1]
            else:
                z_0_prob[i] = (
                    math.log(1 - q) 
                    + math.log(lm_pred_probs[i-1])
                    + np.logaddexp(z_0_prob[i+1], z_1_prob[i+1])
                )

    # Final PPL
    ppl = math.exp((-1 / len(ground_truth)) * z_0_prob[0])

    # Write the result
    with open(f"{trie_ppl_dir}/{k}.json", "w") as f:
        json.dump(ppl, f)


def preprocess_data(args):
    """
    1) Load model & tokenizer
    2) Load your (already chunked) dataset from .pt files (or adapt to chunk on the fly).
    3) Generate hidden states, LM probabilities, ground-truth tokens, and retrieved-chunk JSON.
       So that compute_ppl.py can later compute PPL using them.
    """

    if args.lm_baseline_only:
        base_name = f"eval_{args.dataset}_{args.eval_set}_512_{args.model_ds}_baseline"
    else:
        base_name = f"eval_{args.dataset}_{args.eval_set}_512_DS{args.model_ds}_H{args.model_gen}"

    hidden_states_dir = os.path.join(base_name, "hidden_states")
    if args.save_individual_tries_only:
        chunks_dir = os.path.join(
            base_name,
            f"flattened_retrieved_d{args.partition}_p{args.tok_prob_threshold}_t{args.similarity_threshold}"
        )
    else:
        chunks_dir = os.path.join(
            base_name,
            f"retrieved_d{args.partition}_p{args.tok_prob_threshold}_t{args.similarity_threshold}"
        )

    gt_dir = os.path.join(base_name, "ground_truth")
    lm_pred_probs_dir = os.path.join(base_name, "lm_pred_probs")

    os.makedirs(hidden_states_dir, exist_ok=True)
    os.makedirs(chunks_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(lm_pred_probs_dir, exist_ok=True)

    tokenizer, model = load_model_and_tokenizer(
        model_path=args.model_path, 
        cache_dir=args.cache_dir
    )
    model.eval()
    if torch.cuda.is_available():
        model.to('cuda')

    # We'll create a LMWithTrie instance (as in eval.py).
    # If you want it to do actual retrieval logic,
    # you may need to pass "config" or replicate relevant parts as needed.
    lm_w_trie = LMWithTrie(model, tokenizer, args)

    # TODO: adapt this step to do the chunking
    if args.eval_set == "val":
        dataset_path = f"{args.dataset}_{args.model_gen}_validation_chunked_passages.pt"
    else:
        dataset_path = f"{args.dataset}_{args.model_gen}_test_chunked_passages.pt"

    if not os.path.exists(dataset_path):
        raise ValueError(
            f"Could not find chunked data at {dataset_path}. "
            "Please specify or generate chunked_passages .pt file."
        )

    sentences = torch.load(dataset_path)
    
    # Only process up to n_examples
    n_to_process = min(len(sentences), args.n_examples)

    # ------------------------------------------------------
    # 4) For each chunk, generate hidden_states, LM probs, ground-truth, retrieved chunk
    # ------------------------------------------------------
    # This block follows the "else" part of your eval.py snippet:
    for i, sent in enumerate(sentences):
        if i >= n_to_process:
            break

        # Convert to device
        input_text = sent.to(model.device) if torch.cuda.is_available() else sent

        # 4a) Compute hidden states, LM probabilities if they don't already exist
        hs_exists = os.path.exists(f"{hidden_states_dir}/{i}.pt")
        probs_exists = os.path.exists(f"{lm_pred_probs_dir}/{i}.json")

        if not (hs_exists and probs_exists):
            input_text = input_text.unsqueeze(0)
            output = model(input_text, output_hidden_states=True)
            # TODO: for gpt2 family, no need to do `.squeeze(0)`
            logits = output.logits.squeeze(0)
            hidden_states = output.hidden_states[-1].squeeze(0)

            # compute probabilities
            lm_probs = F.softmax(logits, dim=-1)
            # We skip the first token. So the i-th token prob is lm_probs[i][input_text[i+1]]
            raw_input_text = input_text.squeeze(0)

            lm_pred_probs = []
            for idx, tok in enumerate(raw_input_text[1:]):
                prob = float(lm_probs[idx, tok])
                lm_pred_probs.append(prob)

            with open(f"{lm_pred_probs_dir}/{i}.json", "w") as f:
                json.dump(lm_pred_probs, f)
            save_values(hidden_states_dir, f"{i}.pt", hidden_states)

        # 4b) Retrieve the chunk if not already done
        #    (some code references flatten vs. non-flatten; adapt as needed)
        chunk_json_path = f"{chunks_dir}/{i}.json"
        if not os.path.exists(chunk_json_path):
            # If hidden states weren't just computed above, load them from disk
            # (like eval.py does).
            if hs_exists:
                hidden_states = torch.load(f"{hidden_states_dir}/{i}.pt")
            # run retrieval
            retrieved = lm_w_trie.save_retrieved_chunk(
                hidden_states,
                input_text.to(model.device) if torch.cuda.is_available() else input_text,
                similarity_threshold=args.similarity_threshold
            )
            with open(chunk_json_path, "w") as f:
                json.dump(retrieved, f)

        # 4c) Save the ground-truth if not already
        gt_path = f"{gt_dir}/{i}.pt"
        if not os.path.exists(gt_path):
            # Save the raw tokens (the entire input_text) as ground truth
            save_values(gt_dir, f"{i}.pt", input_text)

def main():
    parser = argparse.ArgumentParser(description="Standalone compute_ppl with preprocessing.")
    # Arguments used for the PPL code
    parser.add_argument("--dataset", type=str, default="pile-of-law-federal_register",
                        help="Dataset name (default: pile-of-law-federal_register)")
    parser.add_argument("--eval_set", type=str, default="test",
                        help="Evaluation split name (default: test)")
    parser.add_argument("--model_ds", type=str, default="some_ds",
                        help="Model DS name or identifier.")
    parser.add_argument("--model_gen", type=str, default="some_gen",
                        help="Model generation name or identifier.")
    parser.add_argument("--q_func", type=str, default="greedy",
                        help="Which q_func to use (default: greedy).")
    parser.add_argument("--q_func_max_prob", type=float, default=1.0,
                        help="Max prob used for 'map' q_func logic (default: 1.0).")
    parser.add_argument("--tok_prob_threshold", type=float, default=0.01,
                        help="Token probability threshold (default: 0.01).")
    parser.add_argument("--similarity_threshold", type=float, default=0.89,
                        help="Similarity threshold (default: 0.89).")
    parser.add_argument("--partition", type=str, default="test",
                        help="Partition name (default: test).")
    parser.add_argument("--n_examples", type=int, default=500,
                        help="Number of examples to process (default: 500).")
    parser.add_argument("--max_workers", type=int, default=8,
                        help="Number of workers for parallel processing (default: 8).")

    # Additional flags controlling behavior
    parser.add_argument("--lm_baseline_only", action="store_true",
                        help="If set, only compute LM baseline PPL (and skip trie-based?).")
    parser.add_argument("--save_individual_tries_only", action="store_true",
                        help="If set, replicate the chunk-dir naming from eval.py's else clause.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path or identifier for the pretrained model.")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Optional huggingface cache dir for the model/tokenizer weights.")

    args = parser.parse_args()

    # --------------------------------------------------------------------------
    # 1) PREPROCESS: Generate hidden states, LM probabilities, retrieved-chunks
    #    so that we can compute PPL. This is the logic previously in eval.py.
    # --------------------------------------------------------------------------
    preprocess_data(args)

    # --------------------------------------------------------------------------
    # 2) Once data are preprocessed, compute perplexities as in original compute_ppl.py
    # --------------------------------------------------------------------------
    if args.lm_baseline_only:
        base_name = f"eval_{args.dataset}_{args.eval_set}_512_{args.model_ds}_baseline"
    else:
        base_name = f"eval_{args.dataset}_{args.eval_set}_512_DS{args.model_ds}_H{args.model_gen}"

    lm_pred_probs_dir = os.path.join(base_name, "lm_pred_probs")
    ppl_results_dir = os.path.join(base_name, "ppl_results")
    os.makedirs(ppl_results_dir, exist_ok=True)

    lm_trie_filename = f"{args.q_func}_d{args.partition}_p{args.tok_prob_threshold}_t{args.similarity_threshold}_max{args.q_func_max_prob}"
    ppl_filename = lm_trie_filename + ".txt"

    trie_pred_probs_dir = os.path.join(base_name, f"lm_trie_pred_probs/{lm_trie_filename}")
    trie_ppl_dir = os.path.join(base_name, f"lm_trie_ppl/{lm_trie_filename}")
    lm_ppl_dir = os.path.join(base_name, "lm_ppl")
    ppl_file_path = os.path.join(ppl_results_dir, ppl_filename)

    os.makedirs(trie_ppl_dir, exist_ok=True)
    os.makedirs(lm_ppl_dir, exist_ok=True)

    ground_truth_dir = os.path.join(base_name, "ground_truth")
    # Flattened or not, we used "flattened_retrieved_..." by default if 
    # save_individual_tries_only was True. Otherwise "retrieved_...".
    # We'll choose the same naming as in preprocess_data:
    if args.save_individual_tries_only:
        retrieved_dir = os.path.join(
            base_name,
            f"flattened_retrieved_d{args.partition}_p{args.tok_prob_threshold}_t{args.similarity_threshold}"
        )
    else:
        retrieved_dir = os.path.join(
            base_name,
            f"retrieved_d{args.partition}_p{args.tok_prob_threshold}_t{args.similarity_threshold}"
        )

    # 2a) If only computing LM baseline, do that in parallel
    if args.lm_baseline_only:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            for k in range(args.n_examples):
                executor.submit(
                    compute_ppl_lm_baseline_only,
                    k,
                    lm_pred_probs_dir,
                    lm_ppl_dir
                )
        # We are done if we want ONLY the LM baseline.
        # If you want to compute the average LM baseline PPL here, do so:
        files = glob.glob(f"{lm_ppl_dir}/*.json")
        if len(files) < args.n_examples:
            print("Warning: some baseline PPL computations are missing.")
        # (Optionally read them and do an average, similar to how we do below)
        return

    # 2b) Otherwise, compute trie-based ppl in parallel
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        for k in range(args.n_examples):
            executor.submit(
                compute_seq_ppl,
                k,
                base_name,
                trie_ppl_dir,
                lm_pred_probs_dir,
                ground_truth_dir,
                retrieved_dir,
                args.dataset,
                args.partition,
                args.q_func,
                args.q_func_max_prob,
                args.tok_prob_threshold,
                args.similarity_threshold
            )

    # 2c) Verify number of outputs
    files = glob.glob(f"{trie_ppl_dir}/*.json")
    if len(files) != args.n_examples:
        print(f"Warning: we expected {args.n_examples} trie-based PPL files, but found {len(files)}.")

    # 2d) Also compute LM baseline perplexities in parallel (if not already done)
    #     so we can compare with trie-based.
    #     If they exist, we won't re-run them.
    existing_baseline = len(glob.glob(f"{lm_ppl_dir}/*.json"))
    if existing_baseline < args.n_examples:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            for k in range(args.n_examples):
                if not os.path.exists(f"{lm_ppl_dir}/{k}.json"):
                    executor.submit(
                        compute_ppl_lm_baseline_only,
                        k,
                        lm_pred_probs_dir,
                        lm_ppl_dir
                    )

    # 2e) Compute average perplexities and save
    trie_ppl_sum = 0
    lm_ppl_sum = 0

    for i in range(args.n_examples):
        # load trie ppl
        trie_path = f"{trie_ppl_dir}/{i}.json"
        lm_path = f"{lm_ppl_dir}/{i}.json"
        if not os.path.exists(trie_path) or not os.path.exists(lm_path):
            print(f"Missing file for example {i}, skipping in final average.")
            continue

        trie_ppl_sum += load_json(trie_path)
        lm_ppl_sum += load_json(lm_path)

    # If all files exist, do the average
    actual_count = min(len(files), args.n_examples)
    if actual_count > 0:
        final_res = {
            "trie_ppl": trie_ppl_sum / actual_count,
            "lm_ppl": lm_ppl_sum / actual_count
        }
    else:
        final_res = {"trie_ppl": None, "lm_ppl": None}

    # 2f) Write the final combined results
    with open(os.path.join(trie_ppl_dir, "ppl_res.json"), "w") as f:
        json.dump(final_res, f)

    print("Done! Results:", final_res)


if __name__ == "__main__":
    main()
