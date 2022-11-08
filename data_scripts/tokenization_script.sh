NSPLIT=128 #Must be larger than the number of processes used during training
FILENAME=en_XX.txt
INFILE=./${FILENAME}
TOKENIZER=bert-base-uncased
#TOKENIZER=bert-base-multilingual-cased
SPLITDIR=./tmp-tokenization-${TOKENIZER}-${FILENAME}/
# this cut command is splitting FILENAME on . and taking the first filed (i.e. en_XX)
OUTDIR=./encoded-data/${TOKENIZER}/$(echo "$FILENAME" | cut -f 1 -d '.')
NPROCESS=8

mkdir -p ${SPLITDIR}
echo ${INFILE}
# -a 3 specifies the 3 digits used to label outfiles (e.g. 001)
# -d specifies digit labeling (not alphabetic)
# -n is the number of splits and l/N specifies that we want N splits and don't
# want to split lines
# SPLITDIR is the prefix of all the file chunks
split -a 3 -d -n l/${NSPLIT} ${INFILE} ${SPLITDIR}

pids=()

for ((i=0;i<$NSPLIT;i++)); do
    num=$(printf "%03d\n" $i);
    FILE=${SPLITDIR}${num};
    #we used --normalize_text as an additional option for mContriever
    python ../preprocess.py --tokenizer ${TOKENIZER} --datapath ${FILE} --outdir ${OUTDIR} &
    pids+=($!);
    if (( $i % $NPROCESS == 0 ))
    then
        for pid in ${pids[@]}; do
            wait $pid
        done
    fi
done

for pid in ${pids[@]}; do
    wait $pid
done

echo ${SPLITDIR}

rm -r ${SPLITDIR}
