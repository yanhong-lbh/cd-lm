import argparse
import json
import numpy as np
import h5py
from collections import defaultdict
import os
import glob
import time
from concurrent.futures import ProcessPoolExecutor


def process_token_with_hdf5(token_id,
                            save_dir,
                            idx_to_seqlen,
                            index_to_acc_index,
                            all_hidden_states_path,
                            partition,
                            model_gen,
                            model_ds):
    """
    For a given token_id, gather positions from tok_pos_{token_id}.json,
    load the relevant hidden states, and save them in an HDF5 file.
    """
    depth_to_hidden_states = defaultdict(list)

    file_name = f'{save_dir}/{token_id}.h5'
    if os.path.exists(file_name):
        return None  # Skip if it already exists

    json_file_name = f'{save_dir}/{token_id}.json'
    tok_pos_json_file_name = f'{save_dir}/tok_pos_{token_id}.json'

    with open(tok_pos_json_file_name) as json_file:
        depth_to_tok_pos = json.load(json_file)

    sorted_depths = sorted(depth_to_tok_pos.keys(), reverse=True)

    with h5py.File(all_hidden_states_path, 'r') as f:
        all_hidden_states = f['all_hidden_states']

        for depth in sorted_depths:
            tok_pos_list = depth_to_tok_pos[depth]
            for chunk_id, tok_pos in tok_pos_list:
                # Example: If you have an offset for certain chunk IDs, handle it here
                # The original code had special logic for federal_register + partition == 'test'.
                # That logic is removed or you can replicate as needed.

                seq_len = idx_to_seqlen[str(chunk_id)]
                acc_idx = index_to_acc_index[str(chunk_id)]

                extracted_hidden_states = all_hidden_states[acc_idx + tok_pos].copy()
                depth_to_hidden_states[depth].append(np.array(extracted_hidden_states))

        # Write out the collated hidden states into an HDF5 file
        with h5py.File(file_name, 'w') as out_f:
            hidden_states_group = out_f.create_group("hidden_states")
            for depth, hidden_states_list in depth_to_hidden_states.items():
                hidden_states_group.create_dataset(
                    str(depth),
                    data=np.stack(hidden_states_list)
                )

    print(f"[INFO] Saved {token_id}.h5")


def parallel_processing(token_ids,
                        save_dir,
                        idx_to_seqlen,
                        index_to_acc_index,
                        all_hidden_states_path,
                        partition,
                        model_gen,
                        model_ds,
                        max_workers=32):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for token_id in token_ids:
            future = executor.submit(
                process_token_with_hdf5,
                token_id,
                save_dir,
                idx_to_seqlen,
                index_to_acc_index,
                all_hidden_states_path,
                partition,
                model_gen,
                model_ds
            )
            futures.append(future)

        results = [f.result() for f in futures]
    return results


def main():
    parser = argparse.ArgumentParser(description="Build and save a datastore (token trie) from extracted hidden states.")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Directory where the token trie data (tok_pos_*.json) is stored and *.h5 will be saved.")
    parser.add_argument("--model_ds", type=str, default="gpt2",
                        help="Short label for the model from which token info was derived.")
    parser.add_argument("--model_gen", type=str, default="gpt2",
                        help="Model used to generate the hidden states memmap/h5.")
    parser.add_argument("--partition", type=str, default="train",
                        choices=["train", "validation", "test"],
                        help="Which dataset partition to process.")
    parser.add_argument("--process_missing_tokens", action="store_true",
                        help="If set, process only the missing tokens from a file named `missing_tokens_{threshold}.json`.")
    parser.add_argument("--missing_tokens_file", type=str, default="",
                        help="Path to JSON containing a list of missing tokens. Required if --process_missing_tokens is used.")
    parser.add_argument("--excluded_tries_prior", type=int, nargs="*", default=[],
                        help="IDs of token tries to exclude if partition == 'train'. (Original logic leftover, use if needed.)")
    parser.add_argument("--k", type=int, default=0,
                        help="Index of chunk among the total in a parallel run. (Used for slicing token_ids.)")
    parser.add_argument("--k_interval", type=int, default=100000,
                        help="Interval slice length per job in a parallel run.")
    parser.add_argument("--max_workers", type=int, default=32,
                        help="Number of parallel processes.")
    parser.add_argument("--idx_to_seqlen_path", type=str, required=True,
                        help="Path to JSON mapping chunk_id -> sequence length (generated at memmap step).")
    parser.add_argument("--index_to_acc_index_path", type=str, required=True,
                        help="Path to JSON mapping chunk_id -> accumulated index offset (generated at memmap step).")
    parser.add_argument("--all_hidden_states_path", type=str, required=True,
                        help="Path to the single HDF5 containing all hidden states (merged at memmap step).")

    args = parser.parse_args()

    # ----------------------------------------------------------------------
    # 1. Build final directory for saving .h5 files (if needed)
    # ----------------------------------------------------------------------
    os.makedirs(args.save_dir, exist_ok=True)

    # ----------------------------------------------------------------------
    # 2. Collect relevant token IDs
    # ----------------------------------------------------------------------
    # The script expects files named "tok_pos_{token_id}.json" in save_dir.
    files = glob.glob(f"{args.save_dir}/tok_pos_*.json")
    token_ids = [os.path.basename(path).split('.')[0].split('_')[-1] for path in files]

    # If user wants to process only missing tokens
    if args.process_missing_tokens:
        if not args.missing_tokens_file:
            raise ValueError("You must provide --missing_tokens_file when using --process_missing_tokens.")
        with open(args.missing_tokens_file, "r") as f:
            missing_tokens = json.load(f)
        # Convert any int to str if needed
        missing_tokens = [str(t) for t in missing_tokens]
        token_ids = list(set(token_ids).intersection(set(missing_tokens)))

    # If training partition and user wants to exclude some tries
    if args.partition == 'train' and args.excluded_tries_prior:
        ignored_ids = [str(i) for i in args.excluded_tries_prior]
        token_ids = list(set(token_ids) - set(ignored_ids))

    # ----------------------------------------------------------------------
    # 3. Load indexing metadata (from memmap step)
    # ----------------------------------------------------------------------
    with open(args.idx_to_seqlen_path, "r") as f:
        idx_to_seqlen = json.load(f)

    with open(args.index_to_acc_index_path, "r") as f:
        index_to_acc_index = json.load(f)

    # We only process a slice of token_ids if parallelizing with k/k_interval
    start_idx = args.k * args.k_interval
    end_idx = (args.k + 1) * args.k_interval
    tokens_to_process = token_ids[start_idx:end_idx]

    # ----------------------------------------------------------------------
    # 4. Load the big HDF5 of all hidden states
    # ----------------------------------------------------------------------
    all_hidden_states_path = args.all_hidden_states_path

    # ----------------------------------------------------------------------
    # 5. Process each token_id in parallel
    # ----------------------------------------------------------------------
    start_time = time.time()
    parallel_processing(
        tokens_to_process,
        args.save_dir,
        idx_to_seqlen,
        index_to_acc_index,
        all_hidden_states_path,
        partition=args.partition,
        model_gen=args.model_gen,
        model_ds=args.model_ds,
        max_workers=args.max_workers
    )
    time_elapsed = time.time() - start_time
    print(f"[INFO] Time elapsed: {time_elapsed:.2f} seconds.")


if __name__ == '__main__':
    main()
