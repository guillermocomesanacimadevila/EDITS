#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 directory"
    exit 1
fi

# Assign input argument to a variable
DIRECTORY="$1"

# Check if the directory exists
if [ ! -d "$DIRECTORY" ]; then
    echo "Directory '$DIRECTORY' does not exist."
    exit 1
fi

# Iterate over all the files in the directory
for file in "$DIRECTORY"/*; do
    # Check if it's a file (not a directory)
    if [ -f "$file" ]; then
        # Get the basename of the file
        filename=$(basename "$file")
        # Replace spaces with underscores in the filename
        new_filename=$(echo "$filename" | tr ' ' '_')
        # Rename the file in place
        mv "$file" "$DIRECTORY/$new_filename"
        # Print the renamed filename
        echo "Renamed '$filename' to '$new_filename'"
    fi
done

echo "All files have been processed."
