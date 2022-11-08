# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import argparse
import torch

import transformers
from src.normalize_text import normalize


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

    # I don't like the way they're doing this. Not respecting document boundaries at all
    # TODO: back up a step and think more carefully about how to split wikipedia on complete
    # documents, and then think about how to split  
    # could feed batches of complete documents from the datasets libray and
    # then breakdown by paragraphs (split on \n\n), and then save encoded content
    # as JSON files mapping title_pararagraph_number to the encoded text content
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--datapath", type=str)
    parser.add_argument("--outdir", type=str)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--normalize_text", action="store_true")

    args, _ = parser.parse_known_args()
    tokenize_file(args)
