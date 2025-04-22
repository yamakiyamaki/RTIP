#!/bin/bash

# RUN THIS COMMAND WHEN YOU CREATE THIS .sh FILE
# chmod +x exe_cuda.sh     


PROGRAM=$1
IMAGE=$2
RESULT=$3
ITER=$4
MODE=$5
KSIZE=$6
SIGMA=$7
FACTOR=$8
CHECK=$9

/usr/local/cuda/bin/nvcc -O3 "$PROGRAM".cu `pkg-config opencv4 --cflags --libs` "$PROGRAM".cpp -o "$PROGRAM"
# if [ "$CHECK" = "check" ]; then
#     # ./exe_cuda.sh PW2_Ex1-2_cuda statue result 500 gauss 3.0 0.8
#     cuda-memcheck ./"$PROGRAM" "$IMAGE".jpg "$RESULT".png "$ITER" "$KSIZE" "$SIGMA" "$FACTOR"
#     xdg-open "$RESULT".png
# fi

if [ "$MODE" = "gauss" ]; then
    # ./exe_cuda.sh PW2_Ex1-2_cuda statue result 500 gauss 3.0 0.8
    ./"$PROGRAM" "$IMAGE".jpg "$RESULT".png "$ITER" "$KSIZE" "$SIGMA" "$FACTOR"
    xdg-open "$RESULT".png
else 
    # ./exe_cuda.sh PW2_Ex1-1_cuda statue result 500 true 
    ./"$PROGRAM" "$IMAGE".jpg "$RESULT".png "$ITER" "$MODE" # Anaglyphs
    xdg-open "$RESULT".png
fi

# original execution command
# /usr/local/cuda/bin/nvcc -O3 PW2_Ex1-1_cuda.cu `pkg-config opencv4 --cflags --libs` PW2_Ex1-1_cuda.cpp -o PW2_Ex1-1_cuda
# //  ./PW2_Ex1-1_cuda statue.jpg result.png 500 true
# //  xdg-open result.png