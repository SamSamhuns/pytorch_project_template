#!/bin/bash

echo "Removing all .DS_Store files"
find . -type f -name '*.DS_Store' -delete
find . -type f -name '*._.DS_Store' -delete

echo "Removing python build artifacts"
# + at the end means the cmd is executed for all file/dir with matches
find . -type d -name "__pycache__" -exec rm -r "{}" +

echo "Removing pytorch training experiment log artifacts & logs"
rm -rf experiments*
rm -rf logs*
rm -rf .data_cache
