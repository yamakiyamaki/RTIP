#!/bin/bash

g++ Ex1-1_omp.cpp -fopenmp `pkg-config opencv4 --cflags` -c
g++ Ex1-1_omp.o  -fopenmp `pkg-config opencv4 --libs` -lstdc++ -o Ex1-1_omp

IMAGE=$1
ANAGLYPHS=$2
./Ex1-1_omp "$IMAGE".jpg "$ANAGLYPHS"
