import os
import glob
import json
import pickle
import numpy as np
import h5py
import torch
import torch.nn.functional as F
import progressbar
import threading

class ThreadWithResult(threading.Thread):
    """
    Simple extension of threading.Thread that allows returning results from the thread.
    """
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        super().__init__(group=group, target=target, name=name, args=args, kwargs=kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def result(self):
        return self._return


class LMWithTrie:
    """
    A wrapper around a language model that can optionally retrieve entire token chunks
    from a prebuilt token-trie datastore using hidden-state similarity.
    """

    def __init__(
        self,
        lm,
        tokenizer,
        token_trie_dir,
        dataset,
        partition,
        tok_prob_threshold,
        model_ds,
        model_gen,
        no_reprocessing=False,
        move_trie_to_gpu=False,
        exclude_huge_token_tries=False,
        excluded_tries_prior=None
    ):
        """
        Args:
            lm (nn.Module): The loaded language model (with .model, .lm_head).
            tokenizer (PreTrainedTokenizer): Corresponding tokenizer.
            token_trie_dir (str): Directory containing token trie subfolders.
            dataset (str): Dataset name (for naming consistency).
            partition (str): "train", "validation", or "test".
            tok_prob_threshold (float): Threshold used during trie creation.
            model_ds (str): Model used to build the datastore.
            model_gen (str): Model used for generation.
            no_reprocessing (bool): If True, skip converting .h5 to .pkl for tries.
            move_trie_to_gpu (bool): If True, load all .pkl data to GPU up-front.
            exclude_huge_token_tries (bool): If True, skip retrieval for certain tokens.
            excluded_tries_prior (list): List of token IDs to skip if exclude_huge_token_tries is set.
        """
        self.lm = lm
        self.tokenizer = tokenizer
        self.token_trie_dir = token_trie_dir
        self.dataset = dataset
        self.partition = partition
        self.tok_prob_threshold = tok_prob_threshold
        self.model_ds = model_ds
        self.model_gen = model_gen

        self.no_reprocessing = no_reprocessing
        self.move_trie_to_gpu = move_trie_to_gpu
        self.exclude_huge_token_tries = exclude_huge_token_tries
        if excluded_tries_prior is None:
            excluded_tries_prior = []
        self.excluded_tries_prior = excluded_tries_prior

        # Directory where we store pre-flattened/pickled tries
        self.individual_flattened_dir = (
            f"{self.token_trie_dir}/individual_flattened_tries"
        )

        # File for storing preloaded_tries in one big pickle
        self.pickle_file_individual = (
            f"{self.token_trie_dir}/preloaded_individual_tries.pkl"
        )

        self.preloaded_individual_tries = {}

        # Save/flatten .h5 -> .pkl if not skipping reprocessing
        if not self.no_reprocessing:
            self.save_individual_tries()

        # Optionally preload entire set to GPU
        if self.move_trie_to_gpu:
            if not os.path.exists(self.pickle_file_individual):
                # no big pickle -> read small pickles and create it
                self.preload_individual_tries()
                with open(self.pickle_file_individual, 'wb') as f:
                    pickle.dump(self.preloaded_individual_tries, f)
            else:
                # load from the single big pickle
                with open(self.pickle_file_individual, 'rb') as f:
                    self.preloaded_individual_tries = pickle.load(f)

                # move everything to GPU
                for key, val in self.preloaded_individual_tries.items():
                    val_tensor = torch.tensor(val['all_hidden_states']).to(self.lm.device)
                    self.preloaded_individual_tries[key] = {
                        'all_hidden_states': val_tensor,
                        'all_node_values': val['all_node_values']
                    }

    def save_individual_tries(self):
        """
        Convert each {token_id}.h5 file under token_trie_dir into a flattened .pkl
        for easier on-the-fly retrieval. We skip reprocessing if it already exists.
        """

        os.makedirs(self.individual_flattened_dir, exist_ok=True)

        # The subfolder containing h5 is usually named like:
        # token_trie_{dataset}_DS{model_ds}_H{model_gen}/{partition}_{tok_prob_threshold}
        # e.g.: token_trie_wikitext-103_DSgpt2_Hgpt2/train_0.3
        trie_folder = f"token_trie_{self.dataset}_DS{self.model_ds}_H{self.model_gen}"
        save_dir = os.path.join(self.token_trie_dir, trie_folder, f"{self.partition}_{self.tok_prob_threshold}")

        trie_files = glob.glob(os.path.join(save_dir, "*.h5"))
        total_files = len(trie_files)
        processed_files = 0

        widgets = [
            progressbar.Percentage(),
            progressbar.Bar(marker='=', left='[', right=']'),
            ' ',
            progressbar.SimpleProgress(),
            ' ',
            progressbar.ETA()
        ]
        bar = progressbar.ProgressBar(widgets=widgets, max_value=total_files)

        for file_path in trie_files:
            # h5 files are named like "12345.h5"
            file_name = os.path.basename(file_path)
            token_str = file_name.split('.')[0]

            pickle_file_trie = os.path.join(self.individual_flattened_dir, f"{token_str}.pkl")
            if os.path.exists(pickle_file_trie):
                # skip if already processed
                processed_files += 1
                bar.update(processed_files)
                continue

            json_file_path = os.path.join(save_dir, f"{token_str}.json")
            if not os.path.exists(json_file_path):
                # The matching .json with node values might be missing
                processed_files += 1
                bar.update(processed_files)
                continue

            all_hidden_states = None
            all_node_values = []

            # Load data
            with h5py.File(file_path, 'r') as hf:
                sorted_depths = sorted(hf['hidden_states'].keys(), reverse=True)

                with open(json_file_path, 'r') as jfp:
                    depth_to_node_values = json.load(jfp)

                for depth in sorted_depths:
                    arr = hf[f'hidden_states/{depth}'][()]
                    vals = depth_to_node_values[depth]

                    if all_hidden_states is None:
                        all_hidden_states = arr[:]
                    else:
                        all_hidden_states = np.append(all_hidden_states, arr, axis=0)

                    all_node_values += vals

            # If no data found, skip
            if all_hidden_states is None:
                processed_files += 1
                bar.update(processed_files)
                continue

            preloaded_trie = {
                'all_hidden_states': all_hidden_states,
                'all_node_values': all_node_values
            }

            with open(pickle_file_trie, 'wb') as pf:
                pickle.dump(preloaded_trie, pf)

            processed_files += 1
            bar.update(processed_files)

        bar.finish()

    def preload_individual_tries(self):
        """
        Read each per-token .pkl in `individual_flattened_dir` into memory,
        storing them in `self.preloaded_individual_tries`.
        """
        files = glob.glob(os.path.join(self.individual_flattened_dir, "*.pkl"))

        for fpath in files:
            token_str = os.path.basename(fpath).split('.')[0]
            with open(fpath, 'rb') as pf:
                preloaded_trie = pickle.load(pf)
                self.preloaded_individual_tries[token_str] = preloaded_trie

    def get_loaded_trie(self, token_id):
        """
        Retrieve (in CPU memory or GPU memory) the preloaded trie data for a token_id.
        If not in self.preloaded_individual_tries (GPU mode), load from disk.
        Returns None if no data is found.
        """
        token_str = str(token_id)

        # If we have preloaded everything to GPU:
        if self.move_trie_to_gpu:
            return self.preloaded_individual_tries.get(token_str, None)

        # Otherwise read from the per-token pickle on disk
        pickle_path = os.path.join(self.individual_flattened_dir, f"{token_str}.pkl")
        if not os.path.exists(pickle_path):
            return None

        with open(pickle_path, 'rb') as pf:
            return pickle.load(pf)

    def generate(self, input_tokens, max_length, similarity_threshold=0.5, past_key_values=None):
        """
        Generates text step-by-step, retrieving from the trie if similarity >= threshold;
        otherwise backs off to normal LM sampling for one token.
        If similarity_threshold == 1 => do purely naive LM generation (no retrieval).
        """
        device = self.lm.device
        input_ids = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(device)
        output = input_ids
        found_phrase_tokens_list = []

        # If threshold=1 => pure baseline generation
        if similarity_threshold == 1:
            while output.shape[-1] < max_length:
                hidden_states = self.lm.model(output)[0][:, -1, :].detach()
                lm_logits = self.lm.lm_head(hidden_states)
                probs = F.softmax(lm_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                output = torch.cat((output, next_token), dim=1)
            return output

        # Otherwise do retrieval-based generation
        while output.shape[-1] < max_length:
            input_hidden_state = self.lm.model(output)[0][:, -1, :]
            current_token = output[0, -1].item()

            loaded_trie = self.get_loaded_trie(current_token)
            found_retrieval = False

            if loaded_trie is not None:
                all_hidden_states = loaded_trie['all_hidden_states']
                all_node_values = loaded_trie['all_node_values']

                state_matrix = torch.tensor(all_hidden_states, device=device)

                try:
                    similarities = F.cosine_similarity(input_hidden_state, state_matrix, dim=1)
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    if ("out of memory" in str(e).lower()):
                        # fallback to CPU
                        input_hidden_state = input_hidden_state.to('cpu')
                        state_matrix = state_matrix.to('cpu')
                        similarities = F.cosine_similarity(input_hidden_state, state_matrix, dim=1)
                    else:
                        raise e

                max_index = torch.argmax(similarities)
                max_similarity = similarities[max_index]

                if max_similarity >= similarity_threshold:
                    nearest_neighbor = all_node_values[max_index]
                    found_retrieval = True

                    if self.model_ds in ['llama-2-7b-chat', 'mistral-7b-instruct-v0.2']:
                        # We treat nearest_neighbor as token list
                        chunk_tokens = nearest_neighbor[1:]
                    else:
                        # We treat nearest_neighbor as a string, then re-tokenize
                        chunk_tokens = self.tokenizer.encode(nearest_neighbor)[1:]

                    if len(chunk_tokens) > 0:
                        found_phrase_tokens_list.append(chunk_tokens)
                        chunk_tokens_tensor = torch.tensor(chunk_tokens, dtype=torch.long).unsqueeze(0).to(device)
                        output = torch.cat((output, chunk_tokens_tensor), dim=1)
                    else:
                        found_retrieval = False

            # If no retrieval done, do naive single-token generation
            if not found_retrieval:
                hidden_states = self.lm.model(output)[0][:, -1, :].detach()
                lm_logits = self.lm.lm_head(hidden_states)
                probs = F.softmax(lm_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                output = torch.cat((output, next_token), dim=1)

        return output, found_phrase_tokens_list

    def generate_parallel(self, input_tokens, max_length, similarity_threshold=0.5, past_key_values=None):
        """
        Similar to generate(), but uses a parallel thread for naive LM sampling
        while we do trie lookup, then chooses which result to use.
        """
        device = self.lm.device
        input_ids = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(device)
        output = input_ids
        found_phrase_tokens_list = []

        def run_lm(hidden_state):
            lm_logits = self.lm.lm_head(hidden_state)
            probs = F.softmax(lm_logits, dim=-1)
            return torch.multinomial(probs, num_samples=1)

        def run_retrieval(hidden_state, token_id):
            loaded_trie = self.get_loaded_trie(token_id)
            if loaded_trie is None:
                return None
            all_hidden_states = loaded_trie['all_hidden_states']
            all_node_values = loaded_trie['all_node_values']

            matrix = torch.tensor(all_hidden_states, device=device)
            try:
                sims = F.cosine_similarity(hidden_state, matrix, dim=1)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" in str(e).lower():
                    hidden_state = hidden_state.to('cpu')
                    matrix = matrix.to('cpu')
                    sims = F.cosine_similarity(hidden_state, matrix, dim=1)
                else:
                    raise e

            max_index = torch.argmax(sims)
            max_similarity = sims[max_index]
            if max_similarity >= similarity_threshold:
                nearest_neighbor = all_node_values[max_index]
                chunk_tokens = self.tokenizer.encode(nearest_neighbor)[1:]
                if len(chunk_tokens) > 0:
                    found_phrase_tokens_list.append(chunk_tokens)
                    return torch.tensor(chunk_tokens, dtype=torch.long, device=device).unsqueeze(0)
            return None

        while output.shape[-1] < max_length:
            hidden_states = self.lm.model(output)[0][:, -1, :].detach()

            # Launch LM generation in a thread
            lm_thread = ThreadWithResult(target=run_lm, args=(hidden_states,))
            lm_thread.start()

            # Meanwhile do retrieval
            current_token = output[0, -1].item()
            retrieved = run_retrieval(hidden_states, current_token)

            # Wait for LM thread to complete
            lm_thread.join()
            lm_token = lm_thread.result()

            if retrieved is not None:
                # If retrieval succeeded, append chunk
                output = torch.cat((output, retrieved), dim=1)
            else:
                # Fallback: single LM token
                output = torch.cat((output, lm_token), dim=1)

        return output, found_phrase_tokens_list

    def save_retrieved_chunk(self, input_hidden_states, sentence, similarity_threshold=0.5):
        """
        For each token in 'sentence', tries to retrieve a chunk from the trie if the
        hidden state similarity >= threshold.  Returns a dict: {token_idx: (chunk, sim)}
        or None if none was found. Useful for perplexity or other analysis.
        """
        device = self.lm.device
        all_input_hidden_states = input_hidden_states.to(device)
        input_len = all_input_hidden_states.shape[0]

        found_phrase_tokens_dict = {}

        for i in range(input_len):
            found_phrase_tokens_dict[i] = None
            current_token = sentence[i]

            # Possibly skip if user wants to exclude large tries
            if self.exclude_huge_token_tries and self.partition == 'train':
                if current_token in self.excluded_tries_prior:
                    continue

            # Try to load the relevant trie
            loaded_trie = self.get_loaded_trie(current_token)
            if loaded_trie is None:
                continue

            single_hidden_state = all_input_hidden_states[i].unsqueeze(0)
            all_hidden_states = loaded_trie['all_hidden_states']
            all_node_values = loaded_trie['all_node_values']

            state_matrix = torch.tensor(all_hidden_states, dtype=torch.float32)
            try:
                state_matrix = state_matrix.to(device)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("Memory error; leaving state_matrix on CPU.")
                else:
                    raise e

            try:
                sims = F.cosine_similarity(single_hidden_state, state_matrix, dim=1)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" in str(e).lower():
                    print("OOM error; computing similarity on CPU.")
                    single_hidden_state = single_hidden_state.to('cpu')
                    state_matrix = state_matrix.to('cpu')
                    sims = F.cosine_similarity(single_hidden_state, state_matrix, dim=1)
                else:
                    raise e

            max_index = torch.argmax(sims)
            max_similarity = sims[max_index]
            if max_similarity >= similarity_threshold:
                nearest_neighbor = all_node_values[max_index]
                found_phrase_tokens = self.tokenizer.encode(nearest_neighbor)[1:]
                if len(found_phrase_tokens) > 0:
                    found_phrase_tokens_dict[i] = (found_phrase_tokens, float(max_similarity.item()))

        return found_phrase_tokens_dict
