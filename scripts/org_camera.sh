#!/bin/bash
src_dir=/media/xuanerzh/9C33-6BBD/DCIM/100MSDCF/zoom/*.JPG
dest_base=/home/xuanerzh/Downloads/zoom/
# src_dir=/Users/ceciliazhang/Desktop/spring2018/batch_3-1/*.CR2
# dest_base=/Users/ceciliazhang/GoogleDrive/Research_Projects/Facebook_Summer_2017/sequence2
filesperdir=7
atfile=0
# 5 images to folder 78
atdir=0

declare -a name=("/7.jpg" "/6.jpg" "/5.jpg" "/4.jpg" "/3.jpg" "/2.jpg" "/1.jpg") #"/4.jpg" "/1.jpg" "/5.jpg")
declare -a namer=("/1.jpg" "/2.jpg" "/3.jpg" "/4.jpg" "/5.jpg" "/6.jpg" "/7.jpg")

# declare -a name=("/2.CR2" "/1.CR2") # "/1.CR2" "/5.CR2")
n=0
for file in $src_dir
do
    echo $file
    if ((atfile == 0)); then
        dest_dir=$(printf "$dest_base/%0.4d" $atdir)
        [[ -d $dest_dir ]] || mkdir -p $dest_dir
        sfocal=$(identify -verbose $file | grep FocalLengthIn35mmFilm | awk '{print $2}')
        echo $sfocal
        if (($sfocal < 100)); then
            names=$name; else
            names=$namer;
        fi
    fi

    cp $file $dest_dir${names[$atfile]}

    ((atfile++))
    if ((atfile >= filesperdir)); then
        atfile=0
        ((atdir++))
    fi
done
