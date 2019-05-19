#!/bin/bash
src_base=$1 #[YOUR TRAINING DATA PATH]
sfolder=1
efolder=585

for (( i=sfolder; i<=efolder; i++))
do
    src_dir=$(printf "$src_base%0.5d/" $i)
    echo 'WB' > $src_dir'/wb.txt'
    for j in $src_dir'/'*.ARW; do
        python3 main_wb.py --folder $src_dir --file $j
    done
done
