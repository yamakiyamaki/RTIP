#!/bin/bash
# RUN THIS COMMAND
# chmod +x exe_omp.sh     #(<--Only once)
# ./exe_omp.sh PW2_Ex1-2_omp statue gauss 3.0 0.8

PROGRAM=$1
IMAGE=$2
MODE=$3
KSIZE=$4
SIGMA=$5
CORE=$6


g++ "$PROGRAM".cpp -fopenmp `pkg-config opencv4 --cflags` -c
g++ "$PROGRAM".o  -fopenmp `pkg-config opencv4 --libs` -lstdc++ -o "$PROGRAM"


if [ "$MODE" = "gauss" ]; then 
# ./exe_omp.sh PW2_Ex1-2_omp statue gauss 3.0 0.8
  ./"$PROGRAM" "$IMAGE".jpg "$KSIZE" "$SIGMA"
else
  ./"$PROGRAM" "$IMAGE".jpg "$MODE" "$CORE"
fi