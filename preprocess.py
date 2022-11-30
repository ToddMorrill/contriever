# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import argparse
import torch

import transformers
from src.normalize_text import normalize
from datasets import load_dataset

import json
import itertools
import time

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

# Assume that we accept Wikipedia as list of dictionaries.
# We generate the dataset represented as list of dictionaries
# with the following keys: id, url, title, text.
def tokenize_data(data_dict, idx):
    file_name = "tmp_" + str(idx) + ".json"
    savepath = os.path.join(args.outdir, file_name) 
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
    docs = []
    token = tokenizer.batch_encode_plus(data_dict['text'], add_special_tokens=False)['input_ids']
    for i in range(len(data_dict['text'])):
        doc = {
            'id': data_dict['id'][i],
            'title': data_dict['title'][i],
            'tokens': token[i],
            'text': data_dict['text'][i],
            'url': data_dict['url'][i]
        }
        docs.append(doc)

    data_json = {
        'chunkId': idx, 
        'docs': docs
    }
    
    print(f"Saving at {savepath}...")

    with open(savepath, 'w') as f:
        json.dump(data_json, f)

def apply_tokenize_gen_data(data_dict, tokenizer, normalize_text=False):
    alltokens = []
    lines = []
    doc_ends = []

    # for text in data_dict['text']:
    #     if normalize_text:
    #         text = normalize(text)

    #     # Store each document text in lines
    #     # Track the length of the document to know where to insert special token for the end
    #     lines.append(text)
    #     doc_ends.append(len(text))

    # Insert special token for end of document
    tokens = tokenizer.batch_encode_plus(data_dict['text'], add_special_tokens=False)['input_ids']

    # TODO: I'm not sure if these should be equal, but they're currently not.
    # I would have thought the number of words (sum of lengths of each document) should
    # be equal to the total number of tokens produced.

    return tokens

def split_data(input_data, n_chunks=128):
    print(len(input_data))
    step = int(len(input_data)/n_chunks)

    c = 0
    for i in range(0, n_chunks):
        start = i * step
        end = start + step
        tmp_data = input_data.select(range(start, end))

        print(len(tmp_data['text']))
        if len(tmp_data['text']) == 0:
            break

        tokenize_data(tmp_data, c)
        c += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--datapath", type=str)
    parser.add_argument("--outdir", type=str)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--normalize_text", action="store_true")

    args, _ = parser.parse_known_args()

    data_dict = load_dataset("wikipedia", "20220301.en", split="train")
    print('length of whole wikipedia', len(data_dict))
    start = time.time()
    nArticles = 1280
    temp_datadict = data_dict.select(range(0,nArticles))
    split_data(temp_datadict)
    print(nArticles, 'articles, time taken:', time.time()-start)
    # tokenize_file(args)
