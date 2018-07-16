#!/bin/bash
# iphone2x_both      dslr_10x_both
dest_base=/home/xuanerzh/Downloads/zoom/dslr_10x_both/
sfolder=51
efolder=72

for (( i=sfolder; i<=efolder; i++))
do
    dest_dir=$(printf "$dest_base%0.5d/" $i)
    echo $dest_dir
    num=$(find $dest_dir -maxdepth 1 -name '*.ARW' | wc -l)
    echo $num
    
    python ./main_raw.py --folder $dest_dir --new_w 512
done
