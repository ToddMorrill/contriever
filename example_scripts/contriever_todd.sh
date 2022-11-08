#!/bin/bash
#SBATCH --cpus-per-task=5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:3
##SBATCH --time=72:00:00
#SBATCH --job-name=contriever
#SBATCH --output=./logtrain/%A
##SBATCH --partition=learnlab
##SBATCH --mem=99GB
##SBATCH --signal=USR1@140
#SBATCH --open-mode=append


port=$(shuf -i 15000-16000 -n 1)
TDIR="/home/iron-man/Documents/contriever/data_scripts/encoded-data/bert-base-uncased"
TRAINDATASETS="${TDIR}/en_XX/"

rmin=0.05
rmax=0.5
T=0.05
QSIZE=131072
MOM=0.9995
POOL=average
AUG=delete
PAUG=0.1
LC=0.
mo=bert-base-uncased
mp=none

name=$SLURM_JOB_ID-$POOL-rmin$rmin-rmax$rmax-T$T-$QSIZE-$MOM-$mo-$AUG-$PAUG

srun /home/iron-man/Documents/contriever/.venv/bin/python ../train.py \
        --model_path $mp \
        --sampling_coefficient $LC \
        --retriever_model_id $mo --pooling $POOL \
        --augmentation $AUG --prob_augmentation $PAUG \
        --train_data $TRAINDATASETS --loading_mode split \
        --ratio_min $rmin --ratio_max $rmax --chunk_length 256 \
        --momentum $MOM --queue_size $QSIZE --temperature $T \
        --warmup_steps 20000 --total_steps 500000 --lr 0.00005 \
        --name $name \
        --scheduler linear \
        --optim adamw \
        --per_gpu_batch_size 32 \
        --output_dir ./checkpoints/$name \
        --main_port $port \

