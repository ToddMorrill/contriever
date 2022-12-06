for ((i=0;i<10;i++)); do
    num=$(printf "%03d\n" $i);
    echo ${i}
    if (($i % 8 == 0)) && (($i != 0))
    then
        echo "Waited at ${i}"
    fi
done