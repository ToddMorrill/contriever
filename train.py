# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import time
import torch
import logging
import os

import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler

from src.options import Options
from src import data, beir_utils, slurm, dist_utils, utils
from src import moco, inbatch, simclr, simoco

logger = logging.getLogger(__name__)


def train(opt, model, optimizer, scheduler, step):
    run_stats = utils.WeightedAvgStats()

    tb_logger = utils.init_tb_logger(opt.output_dir)

    logger.info("Data loading")
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        tokenizer = model.module.tokenizer
    else:
        tokenizer = model.tokenizer
    collator = data.Collator(opt=opt)
    train_dataset = data.load_data(opt, tokenizer)
    logger.warning(f"Data loading finished for rank {dist_utils.get_rank()}")

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=True,
        num_workers=opt.num_workers,
        collate_fn=collator,
    )

    epoch = 1

    model.train()
    while step < opt.total_steps:
        train_dataset.generate_offset()

        logger.info(f"Start epoch {epoch}")
        step_times = []
        for i, batch in enumerate(train_dataloader):
            step_start = time.time()
            step += 1

            batch = {
                key: value.to(int(os.environ['LOCAL_RANK'])) if isinstance(
                    value, torch.Tensor) else value
                for key, value in batch.items()
            }
            train_loss, iter_stats = model(**batch, stats_prefix="train")

            train_loss.backward()
            optimizer.step()

            scheduler.step()
            model.zero_grad()

            run_stats.update(iter_stats)

            if step % opt.log_freq == 0:
                log = f"{step} / {opt.total_steps}"
                for k, v in sorted(run_stats.average_stats.items()):
                    log += f" | {k}: {v:.3f}"
                    if tb_logger:
                        tb_logger.add_scalar(k, v, step)
                log += f" | lr: {scheduler.get_last_lr()[0]:0.3g}"
                log += f" | Memory: {torch.cuda.max_memory_allocated()//1e9} GiB"

                logger.info(log)
                run_stats.reset()

            if step % opt.eval_freq == 0:
                if isinstance(model,
                              torch.nn.parallel.DistributedDataParallel):
                    encoder = model.module.get_encoder()
                else:
                    encoder = model.get_encoder()
                eval_model(opt,
                           query_encoder=encoder,
                           doc_encoder=encoder,
                           tokenizer=tokenizer,
                           tb_logger=tb_logger,
                           step=step)

                if dist_utils.is_main():
                    utils.save(model, optimizer, scheduler, step, opt,
                               opt.output_dir, f"lastlog")

                model.train()

            if dist_utils.is_main() and step % opt.save_freq == 0:
                utils.save(model, optimizer, scheduler, step, opt,
                           opt.output_dir, f"step-{step}")

            if step > opt.total_steps:
                break

            step_end = time.time()
            step_times.append(step_end - step_start)
            if dist_utils.is_main() and step % opt.log_freq == 0:
                avg_step_time = sum(step_times) / len(step_times)
                logger.info(
                    f'Average optimization step time: {avg_step_time:.2f} seconds at step: {step}'
                )
        epoch += 1


def eval_model(opt, query_encoder, doc_encoder, tokenizer, tb_logger, step):
    for datasetname in opt.eval_datasets:
        metrics = beir_utils.evaluate_model(
            query_encoder,
            doc_encoder,
            tokenizer,
            dataset=datasetname,
            batch_size=opt.per_gpu_eval_batch_size,
            norm_doc=opt.norm_doc,
            norm_query=opt.norm_query,
            beir_dir=opt.eval_datasets_dir,
            score_function=opt.score_function,
            lower_case=opt.lower_case,
            normalize_text=opt.eval_normalize_text,
        )

        message = []
        if dist_utils.is_main():
            for metric in ["NDCG@10", "Recall@10", "Recall@100"]:
                message.append(
                    f"{datasetname}/{metric}: {metrics[metric]:.2f}")
                if tb_logger is not None:
                    tb_logger.add_scalar(f"{datasetname}/{metric}",
                                         metrics[metric], step)
            logger.info(" | ".join(message))


if __name__ == "__main__":
    logger.info("Start")

    # this is the argparser
    options = Options()
    opt = options.parse()

    torch.manual_seed(opt.seed)
    # slurm.init_distributed_mode(opt)
    # slurm.init_signal_handler()

    # use torchrun to handle multi-gpu training
    # tell torchrun how many GPUs to use (i.e. world size)
    if opt.torchrun:
        dist.init_process_group(backend="nccl")
    directory_exists = os.path.isdir(opt.output_dir)
    if dist.is_initialized():
        dist.barrier()
    os.makedirs(opt.output_dir, exist_ok=True)
    if not directory_exists and dist_utils.is_main():
        options.print_options(opt)
    if dist.is_initialized():
        dist.barrier()

    # TODO: do we need to be using this logger?
    utils.init_logger(opt)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if opt.contrastive_mode == "moco":
        model_class = moco.MoCo
    elif opt.contrastive_mode == "inbatch":
        model_class = inbatch.InBatch
    elif opt.contrastive_mode == "simclr":
        model_class = simclr.SimClr
    elif opt.contrastive_mode == "simoco":
        model_class = simoco.SiMoCo
    else:
        raise ValueError(
            f"contrastive mode: {opt.contrastive_mode} not recognised")

    # try to reload latest model
    model_path = os.path.join(opt.output_dir, "checkpoint", "latest")
    # if no model exists or if reset_params=True
    if (not os.path.islink(model_path)) or opt.reset_params:
        model = model_class(opt)
        model = model.to(int(os.environ['LOCAL_RANK']))
        optimizer, scheduler = utils.set_optim(opt, model)
        step = 0
        logger.info(f"Initialized fresh model with new parameters.")
    # otherwise continue training
    else:
        model, optimizer, scheduler, opt_checkpoint, step = utils.load(
            model_class, model_path, opt, reset_params=False)
        logger.info(f"Model loaded from {model_path} at step: {step}")

    logger.info(utils.get_parameters(model))

    if dist.is_initialized():
        # TODO: handle the of single GPU training (i.e. non torchrun)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            output_device=int(os.environ['LOCAL_RANK']),
            find_unused_parameters=False,
        )
        dist.barrier()

    logger.info("Start training")
    train(opt, model, optimizer, scheduler, step)

    # clean up
    dist.destroy_process_group()