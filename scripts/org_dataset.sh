#!/bin/bash

#src_dir=/Users/ceciliazhang/Downloads/defocus
# des_dir=/Users/ceciliazhang/Documents/test_dhde
# 
src_dir=/Volumes/Zhangs/101CANON_DONE_focal
des_dir=/Volumes/Zhangs/sequence_jpeg_focal

# src_dir=/Users/ceciliazhang/GoogleDrive/Research_Projects/Facebook_Summer_2017/sequence
# des_dir=/Users/ceciliazhang/GoogleDrive/Research_Projects/Facebook_Summer_2017/syn_dof_0206

atfile=0
atdir=90
ext=.jpg
file2=0
# declare -a name=("/3.jpg" "/2.jpg" "/1.jpg") #"/4.jpg" "/1.jpg" "/5.jpg")
n=0
for (( file=atfile; file<=atdir; file++ ))
do
    dest_dir=$(printf "$src_dir/%0.4d/aligned" $file)
    echo $dest_dir

    cp $dest_dir"/1"$ext $des_dir"/small/"$file2$ext
    cp $dest_dir"/1"$ext $des_dir"/medium/"$file2$ext
    cp $dest_dir"/2"$ext $des_dir"/large/"$file2$ext
    cp $(printf "$src_dir/%0.4d/*.txt" $file) $des_dir"/focal/"
    ((file2++))
done
