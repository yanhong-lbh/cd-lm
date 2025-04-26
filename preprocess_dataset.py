import os
import json
import argparse
import re
from tqdm import tqdm
from datasets import load_dataset

def write_json(list_of_texts, output_file):
    """
    Write a list of text entries into a JSON file, 
    with each entry stored as {"text": "..."} in a list.
    """
    data = [{"text": text} for text in list_of_texts]
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def preprocess_wikitext_103(cache_dir, output_dir):
    """
    Download and split WikiText-103 dataset, parse it into 
    train.json, validation.json, and test.json in standard format.
    """
    print("Loading WikiText-103 dataset...")
    raw_datasets = load_dataset('wikitext', 'wikitext-103-v1', cache_dir=cache_dir)
    # Each subset is a list of strings in ['text']
    
    for dataset_type in ["train", "validation", "test"]:
        dataset = raw_datasets[dataset_type]["text"]
        parsed_passages = []
        passage = ""
        heading_pattern = r'^ = (.+[^=]) = \n$'
        heading = ""

        # Simple example parsing (you can replicate your original logic here).
        for line in tqdm(dataset, desc=f"Parsing {dataset_type} split"):
            if line.strip() == "":
                # If you have logic about headings, it can go here
                if passage.strip():
                    parsed_passages.append(passage.strip())
                    passage = ""
                continue
            
            # Check if line is heading
            if re.fullmatch(heading_pattern, line):
                if passage.strip():
                    parsed_passages.append(passage.strip())
                    passage = ""
                heading = line.strip()
                passage += heading + " "
            else:
                passage += line.strip() + " "

        # Save the last passage
        if passage.strip():
            parsed_passages.append(passage.strip())

        output_file = os.path.join(output_dir, f"{dataset_type}.json")
        write_json(parsed_passages, output_file)

def preprocess_dockerfile(cache_dir, output_dir):
    """
    Example function for converting a Dockerfile dataset 
    (stored or downloaded) into standard JSON.
    """
    # Suppose you have your Dockerfile dataset locally or from HF
    # This example logic is intentionally minimal, adapt as needed.
    # e.g., load from a local JSON or from huggingface
    example_file_path = os.path.join(cache_dir, "code_datasets/gpt2-xl_code_Dockerfile_train.json")
    with open(example_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # data is presumably a list of strings or a list of dicts, e.g.:
    # data = [{"text": "..."}] or just a list of strings
    # For demonstration, let's assume it's a list of strings:
    
    n_passage = 10000  # user-specified slicing
    train_passages = data[:n_passage]
    
    # Save to train
    write_json(train_passages, os.path.join(output_dir, "train.json"))
    
    # For validation/test, similarly:
    validation_passages = data[n_passage : n_passage + 300]
    write_json(validation_passages, os.path.join(output_dir, "validation.json"))

    # For test, you can do more sophisticated splitting or just put placeholders
    test_passages = data[n_passage + 300 : n_passage + 600]
    write_json(test_passages, os.path.join(output_dir, "test.json"))

def preprocess_med_instruction(cache_dir, output_dir):
    """
    Example for medical instruction dataset from huggingface:
    Mohammed-Altaf/medical-instruction-100k
    """
    print("Loading Mohammed-Altaf/medical-instruction-100k dataset...")
    raw_datasets = load_dataset("Mohammed-Altaf/medical-instruction-100k", cache_dir=cache_dir)
    
    # The dataset has splits: ['train', 'test'], each has a column 'Conversation'
    # We might do custom splitting of the train into train & validation, e.g. last 5000
    train_data = raw_datasets["train"]["Conversation"][:-5000]
    val_data = raw_datasets["train"]["Conversation"][-5000:]
    test_data = raw_datasets["test"]["Conversation"]
    
    # Convert them to your desired format
    train_passages = []
    for conv in train_data:
        # Original logic:
        # instruction, q_and_a = conv.split('[|Human|]')
        # But you can keep it simpler and store raw text
        train_passages.append(conv.strip())

    val_passages = [conv.strip() for conv in val_data]
    test_passages = [conv.strip() for conv in test_data]

    write_json(train_passages, os.path.join(output_dir, "train.json"))
    write_json(val_passages, os.path.join(output_dir, "validation.json"))
    write_json(test_passages, os.path.join(output_dir, "test.json"))

def preprocess_pile_of_law(cache_dir, output_dir):
    """
    Example for pile-of-law/pile-of-law with subset 'federal_register'
    """
    print("Loading pile-of-law/pile-of-law subset: 'federal_register'...")
    raw_datasets = load_dataset('pile-of-law/pile-of-law', 'federal_register', cache_dir=cache_dir)

    # Suppose we only handle train, validation, test splits:
    train_texts = raw_datasets["train"]["text"]
    val_texts = raw_datasets["validation"]["text"]
    test_texts = raw_datasets["test"]["text"] if "test" in raw_datasets else []

    write_json(train_texts, os.path.join(output_dir, "train.json"))
    write_json(val_texts, os.path.join(output_dir, "validation.json"))
    write_json(test_texts, os.path.join(output_dir, "test.json"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="Name of dataset to process: wikitext-103, Dockerfile, med_instruction, pile-of-law-federal_register, etc.")
    parser.add_argument("--cache_dir", type=str, default="hf_cache",
                        help="Directory to store or find huggingface datasets.")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Where to save the resulting train.json, validation.json, and test.json.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset == "wikitext-103-v1":
        preprocess_wikitext_103(args.cache_dir, args.output_dir)
    elif args.dataset == "Dockerfile":
        preprocess_dockerfile(args.cache_dir, args.output_dir)
    elif args.dataset == "med_instruction":
        preprocess_med_instruction(args.cache_dir, args.output_dir)
    elif args.dataset == "pile-of-law-federal_register":
        preprocess_pile_of_law(args.cache_dir, args.output_dir)
    else:
        raise ValueError(f"Unknown dataset '{args.dataset}'. Please add a corresponding function.")
