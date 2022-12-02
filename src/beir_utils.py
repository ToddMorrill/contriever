# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
from collections import defaultdict
import json
from typing import List, Dict

import numpy as np
import torch
import torch.distributed as dist
import beir.util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch, DenseRetrievalParallelExactSearch
from beir.reranking.models import CrossEncoder
from beir.reranking import Rerank
import datasets
import pandas as pd

import src.dist_utils as dist_utils
from src import normalize_text


class DenseEncoderModel:
    def __init__(
        self,
        query_encoder,
        doc_encoder=None,
        tokenizer=None,
        max_length=512,
        add_special_tokens=True,
        norm_query=False,
        norm_doc=False,
        lower_case=False,
        normalize_text=False,
        **kwargs,
    ):
        self.query_encoder = query_encoder
        self.doc_encoder = doc_encoder
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        self.norm_query = norm_query
        self.norm_doc = norm_doc
        self.lower_case = lower_case
        self.normalize_text = normalize_text

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:

        if dist.is_initialized():
            idx = np.array_split(range(len(queries)), dist.get_world_size())[dist.get_rank()]
        else:
            idx = range(len(queries))

        queries = [queries[i] for i in idx]
        if self.normalize_text:
            queries = [normalize_text.normalize(q) for q in queries]
        if self.lower_case:
            queries = [q.lower() for q in queries]

        allemb = []
        nbatch = (len(queries) - 1) // batch_size + 1
        with torch.no_grad():
            for k in range(nbatch):
                start_idx = k * batch_size
                end_idx = min((k + 1) * batch_size, len(queries))

                qencode = self.tokenizer.batch_encode_plus(
                    queries[start_idx:end_idx],
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    add_special_tokens=self.add_special_tokens,
                    return_tensors="pt",
                )
                qencode = {key: value.to(dist_utils.get_rank()) for key, value in qencode.items()}
                emb = self.query_encoder(**qencode, normalize=self.norm_query)
                allemb.append(emb.cpu())

        allemb = torch.cat(allemb, dim=0)
        allemb = allemb.to(dist_utils.get_rank())
        if dist.is_initialized():
            allemb = dist_utils.varsize_gather_nograd(allemb)
        allemb = allemb.cpu().numpy()
        if 'convert_to_tensor' in kwargs and kwargs['convert_to_tensor']:
            return torch.tensor(allemb)
        return allemb

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs):

        if dist.is_initialized():
            idx = np.array_split(range(len(corpus)), dist.get_world_size())[dist.get_rank()]
        else:
            idx = range(len(corpus))
        corpus = [corpus[i] for i in idx]
        corpus = [c["title"] + " " + c["text"] if len(c["title"]) > 0 else c["text"] for c in corpus]
        if self.normalize_text:
            corpus = [normalize_text.normalize(c) for c in corpus]
        if self.lower_case:
            corpus = [c.lower() for c in corpus]

        allemb = []
        nbatch = (len(corpus) - 1) // batch_size + 1
        with torch.no_grad():
            for k in range(nbatch):
                start_idx = k * batch_size
                end_idx = min((k + 1) * batch_size, len(corpus))

                cencode = self.tokenizer.batch_encode_plus(
                    corpus[start_idx:end_idx],
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    add_special_tokens=self.add_special_tokens,
                    return_tensors="pt",
                )
                cencode = {key: value.to(dist_utils.get_rank()) for key, value in cencode.items()}
                emb = self.doc_encoder(**cencode, normalize=self.norm_doc)
                allemb.append(emb.cpu())

        allemb = torch.cat(allemb, dim=0)
        allemb = allemb.to(dist_utils.get_rank())
        if dist.is_initialized():
            allemb = dist_utils.varsize_gather_nograd(allemb)
        allemb = allemb.cpu().numpy()
        if 'convert_to_tensor' in kwargs and kwargs['convert_to_tensor']:
            return torch.tensor(allemb)
        return allemb


def evaluate_model(
    query_encoder,
    doc_encoder,
    tokenizer,
    dataset,
    batch_size=128,
    add_special_tokens=True,
    norm_query=False,
    norm_doc=False,
    is_main=True,
    split="test",
    score_function="dot",
    beir_dir="BEIR/datasets",
    save_results_path=None,
    lower_case=False,
    normalize_text=False,
):

    metrics = defaultdict(list)  # store final results

    if hasattr(query_encoder, "module"):
        query_encoder = query_encoder.module
    query_encoder.eval()

    if doc_encoder is not None:
        if hasattr(doc_encoder, "module"):
            doc_encoder = doc_encoder.module
        doc_encoder.eval()
    else:
        doc_encoder = query_encoder

    # parallel GPU search only works with SBERT model
    # DenseRetrievalParallelExactSearch
    dmodel = DenseRetrievalExactSearch(
        DenseEncoderModel(
            query_encoder=query_encoder,
            doc_encoder=doc_encoder,
            tokenizer=tokenizer,
            add_special_tokens=add_special_tokens,
            norm_query=norm_query,
            norm_doc=norm_doc,
            lower_case=lower_case,
            normalize_text=normalize_text,
        ),
        batch_size=batch_size,
        target_devices=None, # will detect all GPUs available
    )
    # encountered a bug here when trying to sort the corpus with more than one worker
    # https://github.com/beir-cellar/beir/blob/c3334fd5b336dba03c5e3e605a82fcfb1bdf667d/beir/retrieval/search/dense/exact_search_multi_gpu.py#L105
    dmodel.sort_corpus = False
    retriever = EvaluateRetrieval(dmodel, score_function=score_function)
    data_path = os.path.join(beir_dir, dataset)
    if not os.path.isdir(data_path) and is_main:
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
        data_path = beir.util.download_and_unzip(url, beir_dir)
    dist_utils.barrier()

    if not dataset == "cqadupstack":
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
        # convert to arrow dataset
        # queries_df = pd.DataFrame.from_dict(queries, orient='index')
        # queries_df.index.name = 'id'
        # queries_df.reset_index(inplace=True)
        # queries_df.rename(columns={0:'text'}, inplace=True)
        # queries = datasets.Dataset.from_pandas(queries_df)

        # corpus_df = pd.DataFrame.from_dict(corpus, orient='index')
        # corpus_df.index.name = 'id'
        # corpus_df.reset_index(inplace=True)
        # corpus = datasets.Dataset.from_pandas(corpus_df)
        # not sure why we can't use multiple workers here
        # corpus = corpus.map(lambda x: {'len': len(x.get("title", "") + x.get("text", ""))})
        results = retriever.retrieve(corpus, queries)
        if is_main:
            ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
            for metric in (ndcg, _map, recall, precision, "mrr", "recall_cap", "hole"):
                if isinstance(metric, str):
                    metric = retriever.evaluate_custom(qrels, results, retriever.k_values, metric=metric)
                for key, value in metric.items():
                    metrics[key].append(value)
            if save_results_path is not None:
                torch.save(results, f"{save_results_path}")
    elif dataset == "cqadupstack":  # compute macroaverage over datasets
        paths = glob.glob(data_path)
        for path in paths:
            corpus, queries, qrels = GenericDataLoader(data_folder=data_folder).load(split=split)
            results = retriever.retrieve(corpus, queries)
            if is_main:
                ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
                for metric in (ndcg, _map, recall, precision, "mrr", "recall_cap", "hole"):
                    if isinstance(metric, str):
                        metric = retriever.evaluate_custom(qrels, results, retriever.k_values, metric=metric)
                    for key, value in metric.items():
                        metrics[key].append(value)
        for key, value in metrics.items():
            assert (
                len(value) == 12
            ), f"cqadupstack includes 12 datasets, only {len(value)} values were compute for the {key} metric"

    metrics = {key: 100 * np.mean(value) for key, value in metrics.items()}
    # save metrics as a json object
    if save_results_path:
        head, tail = os.path.split(save_results_path)
        metrics_filepath = os.path.join(head, f'{dataset}_metrics.json')
        with open(metrics_filepath, 'w') as f:
            json.dump(metrics, f)
    return metrics
