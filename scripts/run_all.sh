#!/bin/bash
# iphone2x_both      dslr_10x_both
dest_base=/home/xuanerzh/Downloads/zoom/dslr_10x_both/
sfolder=1
efolder=10

n=0
for (( i=sfolder; i<=efolder; i++))
do
    if ((atfile == 0)); then
        dest_dir=$(printf "$dest_base%0.5d/" $i)
        echo $dest_dir
    fi
    num=$(find $dest_dir -maxdepth 1 -name '*.JPG' | wc -l)
    echo $num
    python ./main_crop.py --path $dest_dir --filetxt ${dest_base}filename.txt --num $num --ext JPG
    python ./main_align.py --folder $dest_dir --model ECC --rsz 4
    python ./main_raw.py --folder $dest_dir --new_w 512
done

