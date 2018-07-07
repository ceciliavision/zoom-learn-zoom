#!/bin/bash

# bash ./scripts/org_camera.sh 7 dslr

# src_dir=/home/xuanerzh/Downloads/zoom/iphone2x_jpeg/*.jpg
filetype='.JPG'
src_dir=/media/xuanerzh/9C33-6BBD/DCIM/100MSDCF/zoom/*$filetype
dest_base=/home/xuanerzh/Downloads/zoom/dslr_10x_both/
# src_dir=/Users/ceciliazhang/Desktop/spring2018/batch_3-1/*.CR2
# dest_base=/Users/ceciliazhang/GoogleDrive/Research_Projects/sequence2
filesperdir=${1:-7}
devicetype=${2:-dslr}
atfile=0

# start folder id
atdir=1

echo "Source dir: " $src_dir
echo "Device type: " $2

# declare -a name=("/2.CR2" "/1.CR2") # "/1.CR2" "/5.CR2")
n=1
for file in $src_dir
do
    echo $file
    if ((atfile == 0)); then
        dest_dir=$(printf "$dest_base/%0.5d" $atdir)
        [[ -d $dest_dir ]] || mkdir -p $dest_dir
        if [ $devicetype = "dslr" ]; then
            sfocal=$(identify -verbose $file | grep FocalLengthIn35mmFilm | awk '{print $2}')
            echo "Focal length 35mm equiv: " $sfocal
            if (($sfocal < 100)); then
                names=("/00007" "/00006" "/00005" "/00004" "/00003" "/00002" "/00001"); else
                names=("/00001" "/00002" "/00003" "/00004" "/00005" "/00006" "/00007");
            fi
        elif [ $devicetype = "iphone" ]; then
            sfocal=$(exifprobe $file | grep FocalLengthIn35mmFilm | awk '{print $9}')
            sfocal="${sfocal/%mm/}"
            echo "Focal length 35mm equiv: " $sfocal
            if (($sfocal < 40)); then
                names=("/00002" "/00001"); else
                names=("/00001" "/00002");
            fi
        else
            echo "Unknown image device"
        fi 
    fi

    if [ $devicetype = "dslr" ]; then
        cp $file $dest_dir${names[$atfile]}$filetype
        cp ${file/JPG/ARW} $dest_dir${names[$atfile]}.ARW
    elif [ $devicetype = "iphone" ]; then
        cp $file $dest_dir${names[$atfile]}$filetype
    else
        echo "Unknown image device"
    fi

    ((atfile++))
    if ((atfile >= filesperdir)); then
        atfile=0
        ((atdir++))
    fi
done
