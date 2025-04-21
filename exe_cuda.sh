#!/bin/bash
# RUN THIS COMMAND
# chmod +x exe_cuda.sh     #(<--Only once)
# ./exe_cuda.sh PW2_Ex1-2_omp statue result 500 true 

PROGRAM=$1
IMAGE=$2
RESULT=$3
ITER=$4
MODE=$5
KSIZE=$6
SIGMA=$7

/usr/local/cuda/bin/nvcc -O3 "$PROGRAM".cu `pkg-config opencv4 --cflags --libs` "$PROGRAM".cpp -o "$PROGRAM"

if [ "$MODE" = "gauss" ]; then
  ./"$PROGRAM" "$IMAGE".jpg "$MODE" "$KSIZE" "$SIGMA"
else
  ./"$PROGRAM" "$IMAGE".jpg "$RESULT".png "$ITER" "$MODE" # Anaglyphs
  xdg-open "$RESULT".png
fi

# original execution command
# /usr/local/cuda/bin/nvcc -O3 PW2_Ex1-1_cuda.cu `pkg-config opencv4 --cflags --libs` PW2_Ex1-1_cuda.cpp -o PW2_Ex1-1_cuda
# //  ./PW2_Ex1-1_cuda statue.jpg result.png 500 true
# //  xdg-open result.png