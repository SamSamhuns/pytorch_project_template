#!/bin/bash

# check if 2 cmd args have been entered
if [ $# -ne 2 ]
 then
   echo "The mode must be specified for train/test"
   echo "eg. \$ bash build_docker.sh -m train"
   exit
fi

# check if -h/--http flag has been entered
while [[ "$#" -gt 0 ]]; do
   case $1 in
       -m|--mode) mode="$2"; shift ;;
       *) echo "Unknown parameter: $1";
 exit 1 ;;
   esac
   shift
done

echo "Starting Docker Container in pytorch $mode mode"

docker build -t pytorch_model --build-arg ENV=$mode .
