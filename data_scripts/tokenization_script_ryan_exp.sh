NSPLIT=128 #Must be larger than the number of processes used during training
TOKENIZER=bert-base-uncased
#TOKENIZER=bert-base-multilingual-cased
OUTDIR=./encoded-data/${TOKENIZER}/wikipedia-out
NPROCESS=2

mkdir -p ${OUTDIR}

pids=()

for ((i=0;i<$NSPLIT;i++)); do
    num=$(printf "%03d\n" $i);
    #we used --normalize_text as an additional option for mContriever
    echo ${i}
    python ../preprocess.py --outdir ${OUTDIR} --tokenizer ${TOKENIZER} --chunk ${i} &
    pids+=($!);
    if (($i % $NPROCESS == 0)) && (($i != 0))
    then
        for pid in ${pids[@]}; do
            wait $pid
        done
    fi
done

for pid in ${pids[@]}; do
    wait $pid
done
