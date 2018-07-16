#!/bin/bash
# iphone2x_both_process      dslr_10x_both_process
src_base=/home/xuanerzh/Downloads/zoom/dslr_10x_both/
dest_base=/home/xuanerzh/Downloads/zoom/dslr_10x_both_process/
sfolder=1
efolder=105

for (( i=sfolder; i<=efolder; i++))
do
    dest_dir=$(printf "$dest_base%0.5d/" $i)
    src_dir=$(printf "$src_base%0.5d/" $i)
    echo $dest_dir 
    num=$(find $dest_dir"rawpng/" -maxdepth 1 -name '*.png' | wc -l)
    echo $num

    python ./main_crop.py --path $dest_dir"rawpng/" --src_path $src_dir --num $num --ext png
    python ./main_crop.py --path $dest_dir"rawpng_ds/" --src_path $src_dir --num $num --ext png
    python ./main_align.py --folder $dest_dir"rawpng/" --model ECC --rsz 3
    python ./main_align.py --folder $dest_dir"rawpng_ds/" --model ECC --rsz 1
done
