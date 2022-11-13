# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import argparse
import torch

import transformers
from src.normalize_text import normalize
from datasets import load_dataset


def save(tensor, split_path):
    if not os.path.exists(os.path.dirname(split_path)):
        os.makedirs(os.path.dirname(split_path))
    with open(split_path, 'wb') as fout:
        torch.save(tensor, fout)

def apply_tokenizer(path, tokenizer, normalize_text=False):
    alltokens = []
    lines = []
    with open(path, "r", encoding="utf-8") as fin:
        for k, line in enumerate(fin):
            if normalize_text:
                line = normalize(line)

            lines.append(line)
            if len(lines) > 1000000:
                tokens = tokenizer.batch_encode_plus(lines, add_special_tokens=False)['input_ids']
                tokens = [torch.tensor(x, dtype=torch.int) for x in tokens]
                alltokens.extend(tokens)
                lines = []

    tokens = tokenizer.batch_encode_plus(lines, add_special_tokens=False)['input_ids']
    tokens = [torch.tensor(x, dtype=torch.int) for x in tokens]
    alltokens.extend(tokens)

    alltokens = torch.cat(alltokens)
    return alltokens

def tokenize_file(args):
    filename = os.path.basename(args.datapath)
    savepath = os.path.join(args.outdir, f"{filename}.pkl") 
    if os.path.exists(savepath):
        if args.overwrite:
            print(f"File {savepath} already exists, overwriting")
        else:
            print(f"File {savepath} already exists, exiting")
            return
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    except:
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=False)
    print(f"Encoding {args.datapath}...")
    tokens = apply_tokenizer(args.datapath, tokenizer, normalize_text=args.normalize_text)

    print(f"Saving at {savepath}...")
    save(tokens, savepath)

# Assume that we download Wikipedia data (loaded on disk).
# We generate the dataset represented as list of dictionaries
# with the following keys: id, url, title, text.
def tokenize_data(data_dict):
    savepath = os.path.join(args.outdir, f"tmp.pkl") 
    if os.path.exists(savepath):
        if args.overwrite:
            print(f"File {savepath} already exists, overwriting")
        else:
            print(f"File {savepath} already exists, exiting")
            return

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    except:
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=False)
    print(f"Encoding generated dataset...")

    # data_dict is currently being tested using a smaller dataset (Rotten Tomatoes)
    tokens = apply_tokenize_gen_data(data_dict, tokenizer, normalize_text=args.normalize_text)
    
    print(f"Saving at {savepath}...")
    save(tokens, savepath)

def apply_tokenize_gen_data(data_dict, tokenizer, normalize_text=False):
    alltokens = []
    lines = []
    doc_ends = []

    for k, doc in enumerate(data_dict):
        text = doc['text']
        if normalize_text:
            text = normalize(text)

        # Store each document text in lines
        # Track the length of the document to know where to insert special token for the end
        lines.append(text)
        doc_ends.append(len(text))

    # Insert special token for end of document
    tokens = tokenizer.batch_encode_plus(lines, add_special_tokens=False)['input_ids']
    for i in range(len(doc_ends)):
        tokens.insert(doc_ends[i]+ i, [-1])

    tokens = [torch.tensor(x, dtype=torch.int) for x in tokens]
    alltokens.extend(tokens)
    alltokens = torch.cat(alltokens)

    # TODO: I'm not sure if these should be equal, but they're currently not.
    # I would have thought the number of words (sum of lengths of each document) should
    # be equal to the total number of tokens produced.
    print('Number of all words: ', sum(doc_ends))
    print('Number of all tokens: ', len(alltokens))

    return alltokens

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--datapath", type=str)
    parser.add_argument("--outdir", type=str)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--normalize_text", action="store_true")

    args, _ = parser.parse_known_args()
    data_dict = load_dataset("rotten_tomatoes", split="train")
    tokenize_data(data_dict)
    # tokenize_file(args)
