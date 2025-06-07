#!/bin/bash

# RUN THIS COMMAND WHEN YOU CREATE THIS .sh FILE
# chmod +x exe_cuda.sh    

# original execution command
# /usr/local/cuda/bin/nvcc -O3 PW2_Ex1-1_cuda.cu `pkg-config opencv4 --cflags --libs` PW2_Ex1-1_cuda.cpp -o PW2_Ex1-1_cuda
# //  ./PW2_Ex1-1_cuda statue.jpg result.png 500 true
# //  xdg-open result.png


PROGRAM=$1
IMAGE=$2
RESULT=$3
ITER=$4
MODE=$5


/usr/local/cuda/bin/nvcc -O3 "$PROGRAM".cu `pkg-config opencv4 --cflags --libs` "$PROGRAM".cpp -o "$PROGRAM"

if [ "$MODE" = "anaglyphs" ]; then # ./exe_cuda.sh PW2_Ex1-1_cuda statue result 500 anaglyphs true 
    ANAGLYPHS=$6
    ./"$PROGRAM" "$IMAGE".jpg "$RESULT".png "$ITER" "$ANAGLYPHS" # Anaglyphs
    xdg-open "$RESULT".png
elif [ "$MODE" = "anaglyphs_opt" ]; then # ./exe_cuda.sh PW2_Ex1-1_cuda statue result 500 anaglyphs_opt true 32 8
    ANAGLYPHS=$6
    BLINES=$7
    BCOLS=$8
    ./"$PROGRAM" "$IMAGE".jpg "$RESULT".png "$ITER" "$ANAGLYPHS" "$BLINES" "$BCOLS"
    xdg-open "$RESULT".png
elif [ "$MODE" = "gauss" ]; then # ./exe_cuda.sh PW2_Ex1-2_cuda statue result 500 gauss 3.0 0.8
    KSIZE=$6
    SIGMA=$7
    ./"$PROGRAM" "$IMAGE".jpg "$RESULT".png "$ITER" "$KSIZE" "$SIGMA"
    xdg-open "$RESULT".png
elif [ "$MODE" = "gauss_neighbor" ]; then # ./exe_cuda.sh PW2_Ex1-3_cuda statue result 500 gauss 3.0 0.8 5
    KSIZE=$6
    SIGMA=$7
    FACTOR=$8
    ./"$PROGRAM" "$IMAGE".jpg "$RESULT".png "$ITER" "$KSIZE" "$SIGMA" "$FACTOR"
    xdg-open "$RESULT".png
fi

