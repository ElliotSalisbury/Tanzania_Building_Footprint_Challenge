#!/bin/bash
# we use the GDAL tool to split up the massive satellite imagery tif files

IN_FOLDER=$1
OUT_FOLDER=$2
OUT_FOLDER_TIF="${OUT_FOLDER}/tif"
OUT_FOLDER_JPG="${OUT_FOLDER}/jpg"
mkdir -p "${OUT_FOLDER_TIF}"
mkdir -p "${OUT_FOLDER_JPG}"

for file in ${IN_FOLDER}/*.tif; do
    FILENAME=${file##*/}
    echo ${FILENAME}

    IN_FILEPATH="${IN_FOLDER}/${FILENAME}"

    SIZE=$(gdalinfo $IN_FILEPATH | grep "Size is" | cut -c 9-)
    IFS=', ' read -r -a array <<< $SIZE
    WIDTH=${array[0]}
    HEIGHT=${array[1]}
    WINSIZE=$3
    WINSTEP=$4

    # loop over every window in the larger_tiff
    W=0
    while (($W < $WIDTH - $WINSIZE)); do
        H=0
        while (($H < $HEIGHT - $WINSIZE)); do
            OUT_FILEPATH_SMALL="${OUT_FOLDER_TIF}/${FILENAME}.${W}_${H}.tif"
            OUT_FILEPATH_SMALL_JPG="${OUT_FOLDER_JPG}/${FILENAME}.${W}_${H}.jpg"
            gdal_translate -srcwin $W $H $WINSIZE $WINSIZE $IN_FILEPATH $OUT_FILEPATH_SMALL
            gdal_translate -srcwin $W $H $WINSIZE $WINSIZE -of JPEG $IN_FILEPATH $OUT_FILEPATH_SMALL_JPG
            let H=H+WINSTEP
        done
        let W=W+WINSTEP
    done
done