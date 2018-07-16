#!/bin/bash
# iphone2x_both      dslr_10x_both
src_base=/home/xuanerzh/Downloads/zoom/dslr_10x_both/
src_base_process=/home/xuanerzh/Downloads/zoom/dslr_10x_both_process/
sfolder=1
efolder=105

for (( i=sfolder; i<=efolder; i++))
do
    dest_dir=$(printf "$src_base%0.5d/" $i)
    echo $dest_dir
    num=$(find $dest_dir -maxdepth 1 -name '*.ARW' | wc -l)
    echo $num
    
    python ./main_raw.py --folder $dest_dir --new_w 512
done

for (( i=sfolder; i<=efolder; i++))
do
    dest_dir=$(printf "$src_base_process%0.5d/" $i)
    echo $dest_dir
    num=$(find $dest_dir -maxdepth 1 -name '*.png' | wc -l)
    echo $num
    
    python ./main_crop.py --path $dest_dir --num $num --subfolder rawpng/ --ext JPG
    python ./main_crop.py --path $dest_dir --num $num --subfolder rawpng_ds/ --ext JPG
    python ./main_align.py --folder $dest_dir"rawpng/" --model ECC --rsz 3
    python ./main_align.py --folder $dest_dir"rawpng_ds/" --model ECC --rsz 1
done

