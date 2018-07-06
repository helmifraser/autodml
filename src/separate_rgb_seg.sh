#!/bin/sh

declare -a cameras=("CameraRGB" "CameraSemSeg")

for folder in ./*; do
  [ -d "${folder}" ] || continue # if not a directory, skip
  dirname="$(basename "${folder}")"

  # for filename in $folder/*CameraRGB*; do
  #   mv $filename $folder/..
  #   # echo $folder
  # done

  for i in "${cameras[@]}"; do
    if [ ! -d "$dirname/$i" ]; then
      mkdir $dirname/$i
    fi

    for filename in $dirname/*$i*; do
      mv $filename $dirname/$i
    done
  done
done
