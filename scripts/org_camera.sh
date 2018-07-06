#!/bin/bash

# argument
# num of files per folder
# device type

# src_dir=/home/xuanerzh/Downloads/zoom/iphone2x_jpeg/*.jpg
filetype='.dng'
src_dir=/home/xuanerzh/Downloads/zoom/iphone2x_raw/*$filetype
dest_base=/home/xuanerzh/Downloads/zoom/iphone2x_raw/
# src_dir=/Users/ceciliazhang/Desktop/spring2018/batch_3-1/*.CR2
# dest_base=/Users/ceciliazhang/GoogleDrive/Research_Projects/sequence2
filesperdir=${1:-7}
devicetype=${2:-dslr}
atfile=0

# start folder id
atdir=1

echo "Source dir: " $src_dir
echo "Device type: " $2 $3

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
                names=("/00007"$filetype "/00006"$filetype "/00005"$filetype "/00004"$filetype "/00003"$filetype "/00002"$filetype "/00001"$filetype); else
                names=("/00001"$filetype "/00002"$filetype "/00003"$filetype "/00004"$filetype "/00005"$filetype "/00006"$filetype "/00007"$filetype);
            fi
        elif [ $devicetype = "iphone" ]; then
            sfocal=$(exifprobe $file | grep FocalLengthIn35mmFilm | awk '{print $9}')
            sfocal="${sfocal/%mm/}"
            echo "Focal length 35mm equiv: " $sfocal
            if (($sfocal < 40)); then
                names=("/00002"$filetype "/00001"$filetype); else
                names=("/00001"$filetype "/00002"$filetype);
            fi
        else
            echo "Unknown image device"
        fi
        
    fi

    cp $file $dest_dir${names[$atfile]}

    ((atfile++))
    if ((atfile >= filesperdir)); then
        atfile=0
        ((atdir++))
    fi
done
