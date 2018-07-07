#!/bin/bash
dest_base=/home/xuanerzh/Downloads/zoom/dslr_10x_both/
sfolder=1
efolder=5

n=0
for (( i=sfolder; i<=efolder; i++))
do
    if ((atfile == 0)); then
        dest_dir=$(printf "$dest_base%0.5d/" $i)
        echo $dest_dir
    fi
    num=$(find $dest_dir -maxdepth 1 -name '*.JPG' | wc -l)
    echo $num
    python ./main_crop.py --path $dest_dir --filetxt ${dest_base}filename.txt --ext JPG
    python ./main_align.py --folder $dest_dir --ext png --rsz 3
done
