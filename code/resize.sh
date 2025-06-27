#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_file> <size>"
    exit 1
fi

# Get the output prefix from the argument
input_file="$1"
size="$2"

# Extract the base name without extension
base_name=$(basename "$input_file" .tif)

parent_dir=$(dirname "$input_file")
# Create temporary directories for split frames and resized frames

mkdir -p "$parent_dir/frames"
mkdir -p "$parent_dir/resized_frames"

tiffsplit "$input_file" "$parent_dir/frames/frame_"

# resize each frame and save it to the resized_frames directory
for file in "$parent_dir"/frames/frame_*.tif; do
    convert "$file" -resize "${size}!" "$parent_dir/resized_frames/$(basename "$file")"
done

convert "$parent_dir/resized_frames/frame_*.tif" "$parent_dir/${base_name}_${size}.tif"

rm -r "$parent_dir/frames" "$parent_dir/resized_frames"

