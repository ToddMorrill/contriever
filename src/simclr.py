import torch
import torch.nn as nn
import transformers
import logging

from src import contriever, dist_utils, utils

logger = logging.getLogger(__name__)


class SimClr(nn.Module):
    def __init__(self, opt, retriever=None, tokenizer=None, out_dim = -1):
        super(SimClr, self).__init__()

        self.opt = opt
        
        # won't do normalization due to the dimension reduction of the last part 
        self.norm_doc = False   # opt.norm_doc
        self.norm_query = False # opt.norm_query
        
        self.label_smoothing = opt.label_smoothing
        if retriever is None or tokenizer is None:
            retriever, tokenizer = self._load_retriever(
                opt.retriever_model_id, pooling=opt.pooling, random_init=opt.random_init
            )
        self.tokenizer = tokenizer
        self.encoder = retriever
        
        # set output dimension
        self.out_dim = out_dim
        self.mlp = nn.Linear(opt.projection_size, 128)

    def _load_retriever(self, model_id, pooling, random_init):
        cfg = utils.load_hf(transformers.AutoConfig, model_id)
        tokenizer = utils.load_hf(transformers.AutoTokenizer, model_id)

        if "xlm" in model_id:
            model_class = contriever.XLMRetriever
        else:
            model_class = contriever.Contriever

        if random_init:
            retriever = model_class(cfg)
        else:
            retriever = utils.load_hf(model_class, model_id)

        if "bert-" in model_id:
            if tokenizer.bos_token_id is None:
                tokenizer.bos_token = "[CLS]"
            if tokenizer.eos_token_id is None:
                tokenizer.eos_token = "[SEP]"

        retriever.config.pooling = pooling

        return retriever, tokenizer

    def get_encoder(self):
        return self.encoder

    def forward(self, q_tokens, q_mask, k_tokens, k_mask, stats_prefix="", iter_stats={}, **kwargs):

        bsz = len(q_tokens)
        labels = torch.arange(0, bsz, dtype=torch.long, device=q_tokens.device)

        qemb = self.encoder(input_ids=q_tokens, attention_mask=q_mask, normalize=self.norm_query)
        kemb = self.encoder(input_ids=k_tokens, attention_mask=k_mask, normalize=self.norm_doc)

        # dimension reduction and normalize text
        qemb = nn.functional.normalize(self.mlp(qemb), axis=-1)
        kemb = nn.functional.normalize(self.mlp(kemb), axis=-1)

        gather_fn = dist_utils.gather

        gather_kemb = gather_fn(kemb)

        labels = labels + dist_utils.get_rank() * len(kemb)

        scores = torch.einsum("id, jd->ij", qemb / self.opt.temperature, gather_kemb)

        loss = torch.nn.functional.cross_entropy(scores, labels, label_smoothing=self.label_smoothing)

        # log stats
        if len(stats_prefix) > 0:
            stats_prefix = stats_prefix + "/"
        iter_stats[f"{stats_prefix}loss"] = (loss.item(), bsz)

        predicted_idx = torch.argmax(scores, dim=-1)
        accuracy = 100 * (predicted_idx == labels).float().mean()
        stdq = torch.std(qemb, dim=0).mean().item()
        stdk = torch.std(kemb, dim=0).mean().item()
        iter_stats[f"{stats_prefix}accuracy"] = (accuracy, bsz)
        iter_stats[f"{stats_prefix}stdq"] = (stdq, bsz)
        iter_stats[f"{stats_prefix}stdk"] = (stdk, bsz)

        return loss, iter_stats
