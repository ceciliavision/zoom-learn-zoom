#!/bin/bash

dest_base=/home/xuanerzh/Downloads/zoom/dslr_10x_both/

for path in $dest_base*/
do
    num=$(find $path"aligned/" -maxdepth 1 -name '*.png' | wc -l)
    echo $path":" $num
done

