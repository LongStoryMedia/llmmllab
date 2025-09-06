#!/bin/bash

# Script to replace original files with their conflicted copies
# For each file ending with "(conflicted copy *", delete the original file and rename the conflicted copy

echo $PWD

find $PWD -name "*conflicted copy*" | while read conflicted_file; do
    # Extract the base name without the "(conflicted copy *" part
    original_file=$(echo "$conflicted_file" | sed -E 's/ \(conflicted copy [^)]*\)//')
    
    # Check if original file exists
    if [ -f "$original_file" ]; then
        echo "Replacing $original_file with $conflicted_file"
        # Remove the original file
        rm "$original_file"
        # Rename the conflicted copy to the original filename
        mv "$conflicted_file" "$original_file"
    else
        echo "Original file $original_file not found, just renaming $conflicted_file"
        # Just rename the conflicted copy
        mv "$conflicted_file" "$original_file"
    fi
done

echo "Conflict resolution complete!"
