#!/bin/bash
dest_base=/home/xuanerzh/Downloads/burst/
sfolder=8
efolder=26

n=0
for (( i=sfolder; i<=efolder; i++))
do
    if ((atfile == 0)); then
        dest_dir=$(printf "$dest_base/%0.5d/" $i)
        echo $dest_dir
    fi
    num=$(find $dest_dir -maxdepth 1 -name '*.JPG' | wc -l)
    echo $num
    
    python average.py --path $dest_dir --type raw
    python average.py --path $dest_dir --type rgb

done
