import torch
from transformers import (
    GPT2Tokenizer, 
    GPT2LMHeadModel, 
    LogitsProcessorList, 
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    RepetitionPenaltyLogitsProcessor,
    MinLengthLogitsProcessor
)
from datasets import load_dataset
import re
import pickle
import os
import json
import h5py
from tqdm import tqdm
import numpy as np
import glob

from config import config
import time

import progressbar
import torch.nn.functional as F

from utils import sizeof_fmt
import sys

import threading

class ThreadWithResult(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}):
        super().__init__(group=group, target=target, name=name, args=args, kwargs=kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def result(self):
        return self._return

class LMWithTrie:
    def __init__(self, lm, tokenizer, config):
        self.lm = lm
        self.tokenizer = tokenizer
        self.config = config
        self.individual_flattened_dir = f'{config.cache_dir}/individual_flattened_{config.dataset}_{config.partition}_{config.tok_prob_threshold}_DS{config.model_ds}_H{config.model_gen}'
        pickle_file_trie = f'{config.cache_dir}/preloaded_tries_{config.dataset}_{config.partition}_{config.tok_prob_threshold}_DS{config.model_ds}_H{config.model_gen}.pkl'
        pickle_file_flattened = f'{config.cache_dir}/preloaded_flattened_tries_{config.dataset}_{config.partition}_{config.tok_prob_threshold}_DS{config.model_ds}_H{config.model_gen}.pkl'
        pickle_file_individual = f'{config.cache_dir}/preloaded_individual_tries_{config.dataset}_{config.partition}_{config.tok_prob_threshold}_DS{config.model_ds}_H{config.model_gen}.pkl'
        self.preloaded_flattened_tries = {}
        self.preloaded_individual_tries = {}

        if config.save_individual_tries_only:
            if not config.no_reprocessing:
                self.save_individual_tries()
            
            if config.move_trie_to_gpu:
                if not os.path.exists(pickle_file_individual):
                    self.preload_individual_tries()
                    with open(pickle_file_individual, 'wb') as f:
                        pickle.dump(self.preloaded_individual_tries, f)
                else:
                    with open(pickle_file_individual, 'rb') as f:
                        self.preloaded_individual_tries = pickle.load(f)
                        self.preloaded_individual_tries = {k: {'all_hidden_states': torch.tensor(v['all_hidden_states']).to('cuda:0'), 'all_node_values': v['all_node_values']} for k, v in self.preloaded_individual_tries.items()}

    def save_individual_tries(self):
        os.makedirs(self.individual_flattened_dir, exist_ok=True)
        save_dir = f'{config.token_trie_dir}/token_trie_{config.dataset}_DS{config.model_ds}_H{config.model_gen}/{config.partition}_{config.tok_prob_threshold}'

        trie_files = glob.glob(f'{save_dir}/*.h5')

        total_files = len(trie_files)
        processed_files = 0

        widgets = [
            progressbar.Percentage(),
            progressbar.Bar(marker='=', left='[', right=']'),
            ' ', progressbar.SimpleProgress(),
            ' ', progressbar.ETA()
        ]

        bar = progressbar.ProgressBar(widgets=widgets, max_value=total_files)

        for file_path in trie_files:
            all_hidden_states = None
            all_node_values = []
            token = int(os.path.basename(file_path).split('.')[0])
            pickle_file_trie = os.path.join(self.individual_flattened_dir, f"{token}.pkl")

            if os.path.exists(pickle_file_trie):
                print(f"skip {token}")
                continue
            preloaded_trie = {}

            json_file_path = f'{save_dir}/{token}.json'

            with h5py.File(file_path, 'r') as f:
                sorted_depths = sorted(f['hidden_states'].keys(), reverse=True)
                with open(json_file_path) as json_file:
                    depth_to_node_values = json.load(json_file)
                for depth in sorted_depths:
                    arr = f[f'hidden_states/{depth}'][()]
                    vals = depth_to_node_values[depth]

                    if config.model_ds == 'gpt2-xl-conversational':
                        indices = []
                        for i in range(len(vals)):
                            if re.search(r'\n\d', vals[i]) or '[PAD]' in vals[i] or '<|endoftext|>' in vals[i]:
                                
                                if '[PAD]' in vals[i] or '<|endoftext|>' in vals[i]:
                                    vals[i].index('<|endoftext|>')
                                indices.append(i)
                        
                        if len(indices) != 0:
                            new_arr = np.reshape(arr, (-1, 1600))
                            filtered_vals = [j for i, j in enumerate(vals) if i not in indices]
                            if len(filtered_vals) == 0:
                                continue
                            filtered_arr = np.delete(new_arr, indices, axis=0)
                            arr = filtered_arr[:]
                            vals = filtered_vals

                    if all_hidden_states is None:
                        all_hidden_states = arr[:]
                    else:
                        all_hidden_states = np.append(all_hidden_states, arr, axis=0)
                        
                    if config.model_ds in ['llama-2-7b-chat', 'mistral-7b-instruct-v0.2']:

                        name = config.model_ds.split('-')[0]
                        json_file_path = f'{save_dir}/{token}_{name}.json'
                        with open(json_file_path) as json_file:
                            depth_to_node_values = json.load(json_file)

                            if config.dataset == 'MTbench' or config.dataset.endswith('80'):

                                vals = self.tokenizer.batch_decode(depth_to_node_values[depth])

                                for i in range(len(vals)):

                                    m = re.search(r'(\d.)|(\n\d)|(\d\d.)', vals[i])
                                    if m:
                                        new_val = self.tokenizer.encode(vals[i][:m.start(0)])[1:]
                                        depth_to_node_values[depth][i] = new_val

                                decoded = self.tokenizer.batch_decode(depth_to_node_values[depth])

                            all_node_values += depth_to_node_values[depth]

                    else:
                        all_node_values += vals

            if all_hidden_states is None:
                continue

            preloaded_trie = {
                'all_hidden_states': all_hidden_states,
                'all_node_values': all_node_values
            }
            
            with open(pickle_file_trie, 'wb') as f:
                pickle.dump(preloaded_trie, f)

            processed_files += 1
            bar.update(processed_files)

        bar.finish()

        return None

    def preload_individual_tries(self):

        files = glob.glob(f"{self.individual_flattened_dir}/*.pkl")
        for f in files:
            token = f.split('/')[-1].split('.')[0]
            with open(f, 'rb') as f:
                preloaded_trie = pickle.load(f)
                self.preloaded_individual_tries[token] = preloaded_trie

    def preload_flattened_tries(self, partition, tok_prob_threshold):

        save_dir = f'{config.token_trie_dir}/token_trie_{config.dataset}_DS{config.model_ds}_H{config.model_gen}/{partition}_{tok_prob_threshold}'
        os.makedirs(save_dir, exist_ok=True)

        trie_files = glob.glob(f'{save_dir}/*.h5')

        total_files = len(trie_files)
        processed_files = 0

        widgets = [
            progressbar.Percentage(),
            progressbar.Bar(marker='=', left='[', right=']'),
            ' ', progressbar.SimpleProgress(),
            ' ', progressbar.ETA()
        ]

        bar = progressbar.ProgressBar(widgets=widgets, max_value=total_files)

        preloaded_tries = {}
        for file_path in trie_files:
            all_hidden_states = None
            all_node_values = []
            try:
                token = int(os.path.basename(file_path).split('.')[0])

                with h5py.File(file_path, 'r') as f:
                    sorted_depths = sorted(f['hidden_states'].keys(), reverse=True)

                    for depth in sorted_depths:
                        if all_hidden_states is None:
                            all_hidden_states = f[f'hidden_states/{depth}'][()]
                        else:
                            all_hidden_states = np.append(all_hidden_states, f[f'hidden_states/{depth}'][()], axis=0)

                json_file_path = f'{save_dir}/{token}.json'
                with open(json_file_path) as json_file:
                    depth_to_node_values = json.load(json_file)
                    for depth in sorted_depths:
                        all_node_values += depth_to_node_values[depth]

                preloaded_tries[token] = {
                    'all_hidden_states': all_hidden_states,
                    'all_node_values': all_node_values
                }

            except Exception:
                print(f"Corrupt file encountered: {file_path}")
                continue

            processed_files += 1
            bar.update(processed_files)

        bar.finish()

        return preloaded_tries

    def preload_tries(self, partition, tok_prob_threshold):
        save_dir = f'{config.token_trie_dir}/token_trie_{config.dataset}_DS{config.model_ds}_H{config.model_gen}/{partition}_{tok_prob_threshold}'
        os.makedirs(save_dir, exist_ok=True)
        trie_files = glob.glob(f'{save_dir}/*.h5')

        total_files = len(trie_files)
        processed_files = 0

        widgets = [
            progressbar.Percentage(),
            progressbar.Bar(marker='=', left='[', right=']'),
            ' ', progressbar.SimpleProgress(),
            ' ', progressbar.ETA()
        ]

        bar = progressbar.ProgressBar(widgets=widgets, max_value=total_files)

        preloaded_tries = {}
        for file_path in trie_files:
            try:
                token = int(os.path.basename(file_path).split('.')[0])

                with h5py.File(file_path, 'r') as f:
                    depth_to_hidden_states = {int(k): f[f'hidden_states/{k}'][()] for k in f['hidden_states'].keys()}

                json_file_path = f'{save_dir}/{token}.json'
                with open(json_file_path) as json_file:
                    depth_to_node_values = json.load(json_file)

                preloaded_tries[token] = {
                    'depth_to_hidden_states': depth_to_hidden_states,
                    'depth_to_node_values': depth_to_node_values,
                    'sorted_depths': sorted(depth_to_hidden_states.keys(), reverse=True)
                }

            except Exception:
                print(f"Corrupt file encountered: {file_path}")
                continue

            processed_files += 1
            bar.update(processed_files)

        bar.finish()

        return preloaded_tries

    def get_loaded_trie(self, curr_tok_key):

        loaded_trie = None

        if config.move_trie_to_gpu:
            if str(curr_tok_key) in self.preloaded_individual_tries.keys():
                loaded_trie = self.preloaded_individual_tries[f"{curr_tok_key}"]
        else:
            individual_flattened_path = os.path.join(self.individual_flattened_dir, f"{curr_tok_key}.pkl")
        
            if curr_tok_key in self.preloaded_flattened_tries or os.path.exists(individual_flattened_path):

                if config.flatten_trie:
                    loaded_trie = self.preloaded_flattened_tries[curr_tok_key]
                elif config.save_individual_tries_only:
                    with open(individual_flattened_path, 'rb') as f: 
                        loaded_trie = pickle.load(f)
        return loaded_trie if loaded_trie is not None else None

    def generate(self, input_text, max_length, similarity_threshold=0.5, past_key_values=None):
        input_tokens = input_text

        input_ids = torch.tensor(input_tokens).unsqueeze(0).to(self.lm.device)

        found_phrase_tokens_list = []

        output = input_ids

        if similarity_threshold == 1:

            if config.resample_baseline:
                if self.config.model_gen in ['gpt2-xl-conversational']:
                    output = self.lm.generate(input_ids, do_sample=True, temperature=0.3, top_p=0.7, top_k=23, repetition_penalty=1.176, max_length=max_length, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id, use_cache=True)
                else:
                    output = self.lm.generate(input_ids, do_sample=True, max_length=max_length)

                return output

            if self.config.model_gen in ['gpt2-xl-conversational']:
                logits_processor = LogitsProcessorList([
                    # MinLengthLogitsProcessor(max_length, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id),
                    RepetitionPenaltyLogitsProcessor(1.176),
                ])
                logits_warper = LogitsProcessorList([
                    TopKLogitsWarper(23),
                    TemperatureLogitsWarper(0.3),
                    TopPLogitsWarper(0.7),
                ])

                while output.shape[-1] < max_length:
                    input_ids = output
                    hidden_states = self.lm.transformer(input_ids)[0][:,-1,:].detach()
                    lm_logits = self.lm.lm_head(hidden_states)
                    next_token_scores = logits_processor(input_ids, lm_logits)
                    next_token_scores = logits_warper(input_ids, next_token_scores)
                    probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
                    lm_result = torch.multinomial(probs, num_samples=1)
                    output = torch.cat((input_ids, lm_result), dim=1)

            elif self.config.model_gen in ['llama-2-7b-chat', 'mistral-7b-instruct-v0.2']:

                while output.shape[-1] < max_length:
                    input_ids = output
                    hidden_states = self.lm.model(input_ids)[0][:,-1,:].detach()
                    lm_logits = self.lm.lm_head(hidden_states)
                    probs = torch.nn.functional.softmax(lm_logits, dim=-1)
                    lm_result = torch.multinomial(probs, num_samples=1)
                    output = torch.cat((input_ids, lm_result), dim=1)

            else:
                while output.shape[-1] < (max_length + len(input_tokens)):
                    lm_output = self.lm(input_ids, output_hidden_states=True, use_cache = True, past_key_values=past_key_values)
                    input_hidden_state = lm_output.hidden_states[-1][:,-1,:]
                    past_key_values = lm_output.past_key_values
                    logits = lm_output.logits
                    next_token = logits[:, -1:].argmax(dim=-1)

                    output = torch.cat((output, next_token), dim=1)
                    input_ids = next_token  # update input_tokens with the generated token
                
            return output

        if self.config.model_gen in ['gpt2-xl-conversational', 'gpt2-xl', 'llama-2-7b-chat', 'mistral-7b-instruct-v0.2', 'gpt2-large-conversational']: 
            max_length = max_length - len(input_tokens)

        matrix_mul_time = 0
        argmax_time = 0

        while output.shape[-1] < (max_length + len(input_tokens)):

            found_node = False
            nearest_neighbor = None

            if self.config.model_gen in ['gpt2-xl-conversational', 'gpt2-large-conversational']:
                lm_output = self.lm(input_ids)
                input_hidden_state = lm_output.hidden_states[-1][:,-1,:]
            elif self.config.model_gen in ['llama-2-7b-chat', 'mistral-7b-instruct-v0.2']:
                input_hidden_state = self.lm.model(input_ids)[0][:,-1,:]
            elif self.config.model_gen in ['gpt2-xl']:
                lm_output = self.lm(input_ids, output_hidden_states=True)
                input_hidden_state = lm_output.hidden_states[-1][:,-1,:]
            else:
                lm_output = self.lm(input_ids, output_hidden_states=True, use_cache=True, past_key_values=past_key_values)
                input_hidden_state = lm_output.hidden_states[-1][:,-1,:]
                past_key_values = lm_output.past_key_values

            current_token = input_ids[0][-1]
            curr_tok_key = int(current_token)

            if config.save_individual_tries_only:

                loaded_trie = self.get_loaded_trie(curr_tok_key)

                if loaded_trie is not None:
                    found_node = True
                    all_hidden_states = loaded_trie['all_hidden_states']
                    all_node_values = loaded_trie['all_node_values']

                    state_matrix = torch.tensor(all_hidden_states)

                    if config.move_trie_to_gpu:
                        s_time = time.time()
                        similarities = F.cosine_similarity(input_hidden_state, state_matrix, dim=1)
                        e_time = time.time()
                        matrix_mul_time += e_time - s_time
                    else:
                        try:
                            state_matrix = state_matrix.to(input_ids.device)
                        except RuntimeError as e:
                            # check if the error is caused by memory issues
                            if "CUDA out of memory" in str(e):
                                print("Memory error occurred, leaving state_matrix on CPU.")
                            else:
                                # if the error is not due to memory issues, raise the exception
                                raise e

                        try:
                            similarities = F.cosine_similarity(input_hidden_state, state_matrix, dim=1)
                        except torch.cuda.OutOfMemoryError:
                            print("Memory error occurred, doing matrix multiplication on CPU.")
                            input_hidden_state = input_hidden_state.to('cpu')
                            state_matrix = state_matrix.to('cpu')
                            similarities = F.cosine_similarity(input_hidden_state, state_matrix, dim=1)
                        except RuntimeError as e:
                            # check if the error is caused by memory issues
                            if "CUDA out of memory" in str(e):
                                print("Memory error occurred, doing matrix multiplication on CPU.")
                                input_hidden_state = input_hidden_state.to('cpu')
                                state_matrix = state_matrix.to('cpu')
                                similarities = F.cosine_similarity(input_hidden_state, state_matrix, dim=1)
                            else:
                                # if the error is not due to memory issues, raise the exception
                                raise e

                    del state_matrix

                    start = time.time()

                    max_index = torch.argmax(similarities)
                    max_similarity = torch.max(similarities)

                    end = time.time()

                    argmax_time += end - start

                    del similarities

                    if max_similarity >= similarity_threshold:
                        nearest_neighbor = all_node_values[max_index]

                        if config.model_ds in ['llama-2-7b-chat', 'mistral-7b-instruct-v0.2'] :
                            found_phrase_tokens = nearest_neighbor[1:]
                            print(found_phrase_tokens)
                            print(self.tokenizer.decode(found_phrase_tokens))
                        else:
                            found_phrase_tokens = self.tokenizer.encode(nearest_neighbor)[1:]

                        if len(found_phrase_tokens) > 0:
                            found_phrase_tokens_list.append(found_phrase_tokens)
                            found_phrase_tokens = torch.tensor(found_phrase_tokens).to(input_ids.device).unsqueeze(0)

                            output = torch.cat((output, found_phrase_tokens), dim=1)  

                            if self.config.model_gen in ['gpt2-xl-conversational', 'gpt2-large-conversational']:
                                input_ids = output
                                # lm_output = self.lm(input_ids)
                                # input_hidden_state = lm_output.hidden_states[-1][:,-1,:]
                                del lm_output
                            elif config.model_gen in ['llama-2-7b-chat', 'mistral-7b-instruct-v0.2']:
                                input_ids = output
                                input_hidden_state = self.lm.model(input_ids)[0][:,-1,:]
                            else:
                                input_ids = found_phrase_tokens
                                lm_output = self.lm(input_ids, output_hidden_states=True, use_cache=True, past_key_values=past_key_values)
                                input_hidden_state = lm_output.hidden_states[-1][:,-1,:]
                                past_key_values = lm_output.past_key_values
                        else:
                            nearest_neighbor = None

            # back off to LM generation if token is not in the trie, or similarity score lower than similarity_threshold
            if (not found_node) or (not nearest_neighbor):

                if self.config.model_gen in ['gpt2-xl-conversational', 'gpt2-large-conversational']:
                    output = self.lm.generate(input_ids, do_sample=True, temperature=0.3, top_p=0.7, top_k=23, repetition_penalty=1.176, max_new_tokens=1, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
                    #output = self.lm.generate(input_ids, do_sample=True, temperature=0.3, top_p=0.7, top_k=23, repetition_penalty=1.176, max_length=(len(input_ids[0])+1), pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
                    input_ids = output
                elif self.config.model_gen in ['llama-2-7b-chat', 'mistral-7b-instruct-v0.2']:
                    output = self.lm.generate(input_ids, do_sample=True, max_new_tokens=1)
                    input_ids = output
                else:
                    logits = lm_output.logits
                    next_token = logits[:, -1:].argmax(dim=-1)

                    output = torch.cat((output, next_token), dim=1)
                    input_ids = next_token  # update input_tokens with the generated token

        print("mul_time: ", matrix_mul_time)
        print("argmax_time: ", argmax_time)

        return output, found_phrase_tokens_list

    def generate_parallel(self, input_text, max_length, similarity_threshold=0.5, past_key_values=None):
        input_tokens = input_text
        input_ids = torch.tensor(input_tokens).unsqueeze(0).to(self.lm.device)

        found_phrase_tokens_list = []

        output = input_ids

        if self.config.model_gen in ['gpt2-xl-conversational', 'gpt2-xl', 'llama-2-7b-chat', 'mistral-7b-instruct-v0.2']: 
            max_length = max_length - len(input_tokens)

        matrix_mul_time = 0
        argmax_time = 0

        def run_lm(hidden_states, input_ids, logits_processor=None, logits_warper=None):
            if self.config.model_gen == 'gpt2-xl-conversational':

                lm_logits = self.lm.lm_head(hidden_states)
                next_token_scores = logits_processor(input_ids, lm_logits)
                next_token_scores = logits_warper(input_ids, next_token_scores)
                probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
                output = torch.multinomial(probs, num_samples=1)
            elif self.config.model_gen in ['llama-2-7b-chat', 'mistral-7b-instruct-v0.2']:
                lm_logits = self.lm.lm_head(hidden_states)
                probs = torch.nn.functional.softmax(lm_logits, dim=-1)
                output = torch.multinomial(probs, num_samples=1)
            
            return output

        def run_retrieval(input_hidden_state, curr_tok_key, timings):
            start_time = time.time()

            found_node = False
            nearest_neighbor = None

            if self.config.save_individual_tries_only:
                # start = time.time()
                loaded_trie = self.get_loaded_trie(curr_tok_key)
                # timings['get_loaded_trie'] += time.time() - start

                if loaded_trie is not None:
                    found_node = True
                    all_hidden_states = loaded_trie['all_hidden_states']
                    all_node_values = loaded_trie['all_node_values']

                    state_matrix = torch.tensor(all_hidden_states)

                    if config.move_trie_to_gpu:
                        # start = time.time()
                        similarities = F.cosine_similarity(input_hidden_state, state_matrix, dim=1)
                        # timings['cosine_similarity'] += time.time() - start

                    # start = time.time()
                    max_index = torch.argmax(similarities)
                    max_similarity = torch.max(similarities)
                    # timings['max_similarity'] += time.time() - start

                    if max_similarity >= similarity_threshold:
                        nearest_neighbor = all_node_values[max_index]

                        if config.model_ds in ['llama-2-7b-chat', 'mistral-7b-instruct-v0.2'] :
                            found_phrase_tokens = nearest_neighbor[1:]
                        else:
                            # start = time.time()
                            found_phrase_tokens = self.tokenizer.encode(nearest_neighbor)[1:]
                            # timings['tokenizer_encode'] += time.time() - start

                        if len(found_phrase_tokens) > 0:
                            found_phrase_tokens_list.append(found_phrase_tokens)
                            # all LM computation are on the same device
                            found_phrase_tokens = torch.tensor(found_phrase_tokens).to('cuda:1').unsqueeze(0)

                            return found_phrase_tokens, timings
                        
                return None, timings

        trie_time = 0

        timings = {'input_hidden_state': 0, 'current_token_processing': 0, 'get_loaded_trie': 0, 'cosine_similarity': 0, 'max_similarity': 0, 'tokenizer_encode': 0, 'output_concat': 0}

        if self.config.model_gen == 'gpt2-xl-conversational':
            logits_processor = LogitsProcessorList([
                RepetitionPenaltyLogitsProcessor(1.176),
            ])

            logits_warper = LogitsProcessorList([
                TopKLogitsWarper(23),
                TemperatureLogitsWarper(0.3),
                TopPLogitsWarper(0.7),
            ])
        else:
            logits_processor = None
            logits_warper = None

        while output.shape[-1] < (max_length + len(input_tokens)):
            input_ids = output

            if self.config.model_gen == 'gpt2-xl-conversational':
                input_hidden_state = self.lm.transformer(input_ids)[0][:,-1,:].detach()
            elif self.config.model_gen in ['llama-2-7b-chat', 'mistral-7b-instruct-v0.2']:
                input_hidden_state = self.lm.model(input_ids)[0][:,-1,:].detach()

            lm_thread = ThreadWithResult(target=run_lm, args=[input_hidden_state, input_ids, logits_processor, logits_warper])
            lm_thread.start()

            current_token = input_ids[0][-1]
            curr_tok_key = int(current_token)

            trie_result, timings = run_retrieval(input_hidden_state.to('cuda:0'), curr_tok_key, timings)

            lm_thread.join()

            if trie_result is not None:
                # start = time.time()
                output = torch.cat((input_ids, trie_result), dim=1)
                # timings['output_concat'] += time.time() - start
                pass
            else:
                lm_result = lm_thread.result()
                output = torch.cat((input_ids, lm_result), dim=1)

        return output, found_phrase_tokens_list


    def save_retrieved_chunk(self, input_hidden_state, sentence, similarity_threshold=0.5):
        # used for perplexity calculation

        input_ids = sentence
        all_input_hidden_state = input_hidden_state.to(self.lm.device)
        input_len = all_input_hidden_state.shape[0]

        found_phrase_tokens_dict = {}

        for i in range(input_len):

            found_phrase_tokens_dict[i] = None

            nearest_neighbor = None

            input_hidden_state = all_input_hidden_state[i].unsqueeze(0)
            current_token = input_ids[i]
            curr_tok_key = int(current_token)

            if config.save_individual_tries_only:

                individual_flattened_path = os.path.join(self.individual_flattened_dir, f"{curr_tok_key}.pkl")

                if curr_tok_key in self.preloaded_flattened_tries or os.path.exists(individual_flattened_path):
                    if config.exclude_huge_token_tries and config.partition == 'train':
                        # do not do retrieval if the token is in excluded token tries priors
                        if curr_tok_key in config.excluded_tries_prior:
                            continue

                    with open(individual_flattened_path, 'rb') as f: 
                        loaded_trie = pickle.load(f)

                    all_hidden_states = loaded_trie['all_hidden_states']
                    all_node_values = loaded_trie['all_node_values']

                    state_matrix = torch.tensor(all_hidden_states)

                    try:
                        state_matrix = state_matrix.to(input_ids.device)
                    except RuntimeError as e:
                        # check if the error is caused by memory issues
                        if "CUDA out of memory" in str(e):
                            print("Memory error occurred, leaving state_matrix on CPU.")
                        else:
                            # if the error is not due to memory issues, raise the exception
                            raise e

                    if state_matrix.shape[0] != len(all_node_values):
                        print(curr_tok_key)
                        continue

                    # similarities = F.cosine_similarity(input_hidden_state, state_matrix, dim=1)

                    try:
                        similarities = F.cosine_similarity(input_hidden_state, state_matrix, dim=1)
                    except torch.cuda.OutOfMemoryError:
                        print("Memory error occurred, doing matrix multiplication on CPU.")
                        input_hidden_state = input_hidden_state.to('cpu')
                        state_matrix = state_matrix.to('cpu')
                        similarities = F.cosine_similarity(input_hidden_state, state_matrix, dim=1)
                    except RuntimeError as e:
                        # check if the error is caused by memory issues
                        if "CUDA out of memory" in str(e):
                            print("Memory error occurred, doing matrix multiplication on CPU.")
                            input_hidden_state = input_hidden_state.to('cpu')
                            state_matrix = state_matrix.to('cpu')
                            similarities = F.cosine_similarity(input_hidden_state, state_matrix, dim=1)
                        else:
                            # if the error is not due to memory issues, raise the exception
                            raise e

                    max_index = torch.argmax(similarities)
                    max_similarity = torch.max(similarities)

                    if max_similarity >= similarity_threshold:
                        nearest_neighbor = all_node_values[max_index]

                        if config.model_ds in ['llama-2-7b-chat', 'mistral-7b-instruct-v0.2']:
                            found_phrase_tokens = nearest_neighbor[1:]
                        else:
                            found_phrase_tokens = self.tokenizer.encode(nearest_neighbor)[1:]

                        if len(found_phrase_tokens) > 0:
                            found_phrase_tokens_dict[i] = (found_phrase_tokens, float(max_similarity.cpu().detach().numpy()))

            else:
                if curr_tok_key in self.preloaded_tries:
                    loaded_trie = self.preloaded_tries[curr_tok_key]
                    depth_to_hidden_states = loaded_trie['depth_to_hidden_states']
                    depth_to_node_values = loaded_trie['depth_to_node_values']
                    sorted_depths = loaded_trie['sorted_depths']

                    for depth in sorted_depths:
                        # Move depth_to_hidden_states[depth] to GPU
                        state_matrix = torch.tensor(depth_to_hidden_states[depth]).to(input_ids.device)

                        # Speed up similarity calculation with vectorized operations
                        similarities = F.cosine_similarity(input_hidden_state, state_matrix, dim=1)

                        max_index = torch.argmax(similarities)
                        max_similarity = torch.max(similarities)

                        if max_similarity >= similarity_threshold:
                            nearest_neighbor = depth_to_node_values[str(depth)][max_index]

                            found_phrase_tokens = self.tokenizer.encode(nearest_neighbor)[1:]

                            if len(found_phrase_tokens) > 0:
                                found_phrase_tokens_dict[i] = (found_phrase_tokens, float(max_similarity.cpu().detach().numpy()))
                            
                            break

        return found_phrase_tokens_dict
