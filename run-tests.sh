#!/bin/bash

## declare an array variable
declare -a versions=("1.4.0" "1.13.1" "2.0.1")

## now loop through the above array
for v in "${versions[@]}"
do
   echo "testing pytorch:${v} ..."
   image_name="anibali/pytorch:${v}-nocuda"
   container_name="pytorch-${v}"
   log_file_name="pytorch-${v}.log"
   docker run --name "${container_name}" --rm -i -t \
    -v ./pytorch-test.py:/app/pytorch-test.py \
    "${image_name}" python3 /app/pytorch-test.py &> "${log_file_name}"
done