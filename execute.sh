#!/bin/bash
PROGRAM=$1
IMAGE=$2
MODE=$3
KSIZE=$4
SIGMA=$5


g++ "$PROGRAM".cpp -fopenmp `pkg-config opencv4 --cflags` -c
g++ "$PROGRAM".o  -fopenmp `pkg-config opencv4 --libs` -lstdc++ -o "$PROGRAM"


if [ "$MODE" = "gauss" ]; then
  ./"$PROGRAM" "$IMAGE".jpg "$MODE" "$KSIZE" "$SIGMA"
else
  ./"$PROGRAM" "$IMAGE".jpg "$MODE"
fi