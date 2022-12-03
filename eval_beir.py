# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""This module is responsible for evaluating the model's performance on the
BEIR benchmark. 

Examples:
    $ python eval_beir.py \
        --datasets scifact trec-covid nfcorpus scidocs fiqa arguana quora nq hotpotqa dbpedia-entity fever climate-fever msmarco cqadupstack\
        --beir_dir ./beir-data \
        --per_gpu_batch_size 128 \
        --output_dir ./results/v1 \
        --model_name_or_path ./checkpoints/checkpoint/latest

    $ python eval_beir.py \
        --datasets scifact \
        --beir_dir ../contriever/beir-data \
        --per_gpu_batch_size 256 \
        --output_dir ./results/v1 \
        --model_name_or_path ./checkpoints/checkpoint/latest
TODO: debug Touche-2020
"""

import sys
import argparse
import torch
import logging
import json
import numpy as np
import os
import time

import src.slurm
import src.contriever
import src.beir_utils
import src.utils
import src.dist_utils
import src.contriever

logger = logging.getLogger(__name__)


def main(args):

    # src.slurm.init_distributed_mode(args)
    # src.slurm.init_signal_handler()

    os.makedirs(args.output_dir, exist_ok=True)

    logger = src.utils.init_logger(args)

    model, tokenizer, _ = src.contriever.load_retriever(args.model_name_or_path)
    model = model.cuda()
    model.eval()
    query_encoder = model
    doc_encoder = model

    for dataset in args.datasets:
        logger.info(f"Start indexing dataset: {dataset}")
        save_results_path = os.path.join(args.output_dir, f'{dataset}_results.pt')
        # if the file is already there, we've already computed the results, continue
        if os.path.exists(save_results_path):
            continue
        start = time.time()
        metrics = src.beir_utils.evaluate_model(
            query_encoder=query_encoder,
            doc_encoder=doc_encoder,
            tokenizer=tokenizer,
            dataset=dataset,
            batch_size=args.per_gpu_batch_size,
            norm_query=args.norm_query,
            norm_doc=args.norm_doc,
            is_main=src.dist_utils.is_main(),
            split="dev" if dataset == "msmarco" else "test",
            score_function=args.score_function,
            beir_dir=args.beir_dir,
            save_results_path=save_results_path,
            lower_case=args.lower_case,
            normalize_text=args.normalize_text,
        )

        if src.dist_utils.is_main():
            for key, value in metrics.items():
                logger.info(f"{dataset} : {key}: {value:.1f}")

        end = time.time()
        duration = end - start
        logger.info(f'Time taken to evaluate the {dataset} dataset: {duration:.2f} seconds')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--datasets", nargs="*", type=str, help="Evaluation dataset(s) from the BEIR benchmark")
    parser.add_argument("--beir_dir", type=str, default="./", help="Directory to save and load beir datasets")
    parser.add_argument("--text_maxlength", type=int, default=512, help="Maximum text length")

    parser.add_argument("--per_gpu_batch_size", default=128, type=int, help="Batch size per GPU/CPU for indexing.")
    parser.add_argument("--output_dir", type=str, default="./my_experiment", help="Output directory")
    parser.add_argument("--model_name_or_path", type=str, help="Model name or path")
    parser.add_argument(
        "--score_function", type=str, default="dot", help="Metric used to compute similarity between two embeddings"
    )
    parser.add_argument("--norm_query", action="store_true", help="Normalize query representation")
    parser.add_argument("--norm_doc", action="store_true", help="Normalize document representation")
    parser.add_argument("--lower_case", action="store_true", help="lowercase query and document text")
    parser.add_argument(
        "--normalize_text", action="store_true", help="Apply function to normalize some common characters"
    )
    parser.add_argument("--save_results_path", type=str, default=None, help="Path to save result object")

    # parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    # parser.add_argument("--main_port", type=int, default=-1, help="Main port (for multi-node SLURM jobs)")

    args, _ = parser.parse_known_args()
    main(args)
