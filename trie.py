from collections import Counter, defaultdict
import json
import numpy as np
import os
from scipy.spatial.distance import cosine
import h5py
import concurrent.futures
from transformers import GPT2Tokenizer


class TrieNode:
    def __init__(self, value=None, depth=0, token_count=0, parent=None):
        self.value = value
        self.children = {}
        self.is_end_of_phrase = False
        self.id = None
        self.token_prob = {}
        self.token_count = token_count
        self.depth = depth
        self.parent = parent
        self.token_id = None
        self.chunk_len = 0


class Trie:
    def __init__(
        self,
        token_trie_dir=None,
        dataset=None,
        model_ds=None,
        model_gen=None,
        partition=None,
        tok_prob_threshold=0.3
    ):
        """
        Constructor without a global config. Pass in paths/identifiers for naming.

        Parameters:
          token_trie_dir (str): Base directory to save the trie files.
          dataset (str): Dataset name (e.g. 'wikitext-103').
          model_ds (str): Short label for the model from which token probs derived (e.g. 'gpt2').
          model_gen (str): Model used in generating hidden states (e.g. 'gpt2-xl').
          partition (str): 'train', 'validation', or 'test'.
          tok_prob_threshold (float): Probability threshold used in building the trie (for naming).
        """
        self.root = TrieNode()
        self.total_leaves = 0
        self.total_depth = 0
        self.total_nodes = 0

        # Store these for use in set_token_ids() and save_token_trie_info()
        self.token_trie_dir = token_trie_dir
        self.dataset = dataset
        self.model_ds = model_ds
        self.model_gen = model_gen
        self.partition = partition
        self.tok_prob_threshold = tok_prob_threshold

    def insert(self, tokens, uid, token_probs):
        """
        Insert a sequence of tokens into the trie.

        tokens: A list of tokens (strings).
        uid: An integer or unique ID for the final node (leaf).
        token_probs: List of (token, prob, chunk_id, tok_pos_in_chunk) tuples or similar.
                     We only store the first token's chunk/position info in node.token_prob for retrieval.
        """
        node = self.root
        current_depth = 0

        for (token, token_prob) in zip(tokens, token_probs):
            current_depth += 1
            if token not in node.children:
                node.children[token] = TrieNode(value=token, depth=node.depth + 1, parent=node)
            node = node.children[token]
            node.token_count += 1

        node.is_end_of_phrase = True
        node.chunk_len = len(tokens)

        # For convenience, store the chunk ID and position of the first contiguous token
        first_cont_token = tokens[0]
        first_cont_token_prob = token_probs[0]  # (token, prob, chunk_id, tok_pos)
        if first_cont_token in node.token_prob:
            node.token_prob[first_cont_token].append(first_cont_token_prob[1:])
        else:
            node.token_prob[first_cont_token] = [first_cont_token_prob[1:]]

        node.id = uid
        self.total_nodes += current_depth
        self.total_depth += current_depth
        self.total_leaves += 1

    def search(self, prefix_tokens):
        """
        Search the trie for a list of prefix tokens. Returns a list of node IDs for any
        complete phrase that starts with prefix_tokens.
        """
        node = self.root
        for token in prefix_tokens:
            if token not in node.children:
                return []
            node = node.children[token]
        return self._dfs(node)

    def _dfs(self, node):
        """
        Depth-first traversal from a given node, collecting all IDs at end-of-phrase nodes.
        """
        result = []
        if node.is_end_of_phrase:
            result.append(node.id)
        for child in node.children.values():
            result.extend(self._dfs(child))
        return result

    def traverse(self, callback, node=None, prefix=None):
        """
        Generic traversal that calls 'callback(node)' on each node in the trie.
        """
        if node is None:
            node = self.root
            prefix = []

        if node.value is not None:
            prefix.append(node.value)

        callback(node)

        for child_node in node.children.values():
            self.traverse(callback, child_node, prefix.copy())

    def update_depth(self, node, difference):
        """
        Recursively update the depth of 'node' and its children by 'difference'.
        """
        node.depth += difference
        for child in node.children.values():
            self.update_depth(child, difference)

    def compress(self, node=None):
        """
        Compress chains of single-child nodes into a single node by merging their token values.
        """
        if node is None:
            node = self.root

        # If node has exactly one child and node is not root
        if len(node.children) == 1 and node.value is not None:
            child = list(node.children.values())[0]

            # Merge child's value into current node
            node.value += child.value

            # Merge child's token_prob dict
            for token, prob in child.token_prob.items():
                if child.value in node.token_prob:
                    node.token_prob[child.value] += prob
                else:
                    node.token_prob[child.value] = prob

            # If child is an end_of_phrase, reflect that in the parent
            if child.is_end_of_phrase:
                node.is_end_of_phrase = True
                node.id = child.id

            # "Delete" the child, keep child's children
            node.children = child.children

            # Adjust the depth of sub-children
            self.update_depth(node, -1)

            # Continue compressing from this updated node
            self.compress(node)
        else:
            for child in node.children.values():
                self.compress(child)

    def reassign_ids(self, node=None, new_id=0):
        """
        Assign new IDs in a DFS order. Returns the next available ID after finishing this subtree.
        """
        if node is None:
            node = self.root

        node.id = new_id
        new_id += 1

        for child in node.children.values():
            new_id = self.reassign_ids(child, new_id)

        return new_id

    def set_token_ids(self, tokenizer):
        """
        Assign a 'token_id' from the tokenizer to each direct child of the root.
        This snippet checks if the model is 'llama' or 'mistral' to decide which
        position in the encoded result to use. Adjust as needed.
        """
        for child in self.root.children.values():
            child_value = child.value
            tokens = tokenizer.encode(child_value, return_tensors="pt")
            # If your model tends to produce a leading special token for certain models,
            # you can handle that logic here:
            if self.model_ds and (self.model_ds.startswith('llama') or self.model_ds.startswith('mistral')):
                first_token = tokens[0][-1].item()
            else:
                first_token = tokens[0][0].item()

            child.token_id = first_token

    def reassign_values(self, node=None, value=None):
        """
        Reconstruct 'value' by concatenating the parent's value with the child's.
        This is used after compress() or if you need continuous token strings in a node.
        """
        if node is None:
            node = self.root

        if node.parent and (node.parent.value is not None):
            node.value = node.parent.value + node.value

        for child in node.children.values():
            self.reassign_values(child, node.value)

        return node.value

    def save_token_trie_info(self):
        """
        Saves each direct child of root to a JSON file containing the node values,
        plus a second JSON file mapping depths -> chunk positions for retrieving hidden states.

        The final directory structure is:
            {self.token_trie_dir}/token_trie_{dataset}_DS{model_ds}_H{model_gen}/{partition}_{tok_prob_threshold}/
                ├─ {token_id}.json
                └─ tok_pos_{token_id}.json
        """
        save_dir = (
            f"{self.token_trie_dir}/token_trie_{self.dataset}_DS{self.model_ds}_H{self.model_gen}/"
            f"{self.partition}_{self.tok_prob_threshold}"
        )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for child in self.root.children.values():
            token_id = child.token_id
            json_file_name = f"{save_dir}/{token_id}.json"
            tok_pos_json_file_name = f"{save_dir}/tok_pos_{token_id}.json"

            # If both files already exist, skip
            if os.path.exists(json_file_name) and os.path.exists(tok_pos_json_file_name):
                continue

            depth_to_tok_pos, depth_to_node_values = collect_token_info_by_depth(child)

            with open(tok_pos_json_file_name, 'w') as f_tok_pos:
                json.dump(depth_to_tok_pos, f_tok_pos)
            with open(json_file_name, 'w') as f_node_vals:
                json.dump(depth_to_node_values, f_node_vals)


def collect_token_info_by_depth(node):
    """
    Traverse the subtree starting at `node` and collect chunk positions (chunk_id, tok_pos) 
    along with the node's string value. Group them by depth in two dicts:
      - depth_to_tok_pos: { depth -> [(chunk_id, tok_pos), ...] }
      - depth_to_node_values: { depth -> [node_value, ...] }
    """
    depth_to_tok_pos = defaultdict(list)
    depth_to_node_values = defaultdict(list)

    def collect_token_info(current_node):
        # current_node.token_prob structure example: { 'the': [(prob, chunk_id, tok_pos), ...], ... }
        if current_node.is_end_of_phrase:
            for token, prob_list in current_node.token_prob.items():
                for p in prob_list:  # p is (prob, chunk_id, tok_pos) or similar
                    chunk_id = p[1]
                    tok_pos = p[2]
                    depth_to_tok_pos[current_node.depth].append((chunk_id, tok_pos))
                    depth_to_node_values[current_node.depth].append(current_node.value)

        for child_node in current_node.children.values():
            collect_token_info(child_node)

    collect_token_info(node)
    return dict(depth_to_tok_pos), dict(depth_to_node_values)
