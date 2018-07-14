#!/bin/bash

# dslr_10x_both    iphone2x_both

src_dir=/home/xuanerzh/Downloads/zoom/dslr_10x_both/

echo "Source dir: " $src_dir
for path in $src_dir*/
do
    for file in $path*.JPG; do
        echo $file
        exp=$(exifprobe $file | grep '\s\s ExposureTime' | awk '{print $5}')
        iso=$(exifprobe $file | grep 'ISO' | awk '{print $9}')
        filename="${file##*/}"
        echo ${filename%.*}":$exp" >> $path/exp.txt
        echo ${filename%.*}":$iso" >> $path/iso.txt
    done
done
