#!/bin/bash
dest_base=/home/xuanerzh/Downloads/zoom/
sfolder=0
efolder=23

n=0
for (( i=sfolder; i<=efolder; i++))
do
    if ((atfile == 0)); then
        dest_dir=$(printf "$dest_base/%0.5d/" $i)
        echo $dest_dir
    fi
    num=$(find $dest_dir -maxdepth 1 -name '*.jpg' | wc -l)
    echo $num
    python ../template_match.py --path $dest_dir --filetxt /home/xuanerzh/Downloads/zoom/filename.txt 
    # python ../align.py --folder $dest_dir/ --ext '.jpg' --filetxt ../filename.txt --n $num
done
