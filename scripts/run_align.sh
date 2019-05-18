#!/bin/bash

dest_base=/data/zoom/train/ # [YOUR TRAINING DATA PATH]
sfolder=1
efolder=585

for (( i=sfolder; i<=efolder; i++))
do
    dest_dir=$(printf "$dest_base%0.5d/" $i)
    echo $dest_dir 
    num=$(find $dest_dir -maxdepth 1 -name '*.JPG' | wc -l)
    echo $num $dest_dir

    rm -rf $dest_dir/cropped
    rm -rf $dest_dir/aligned
    rm -rf $dest_dir/compare
    
    python3 ./main_crop.py --path $dest_dir --num $num
    python3 ./main_align_camera.py --path $dest_dir --model ECC --rsz 3
done
