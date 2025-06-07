#!/bin/bash

PROGRAM=$1
IMAGE=$2
RESULT=$3
ITER=$4
MODE=$5


# Create results directory if it doesn't exist
mkdir -p benchResults

# Clear previous result file
OUTPUT_FILE="./benchResults/$PROGRAM.txt"
echo "=== Benchmark Results ===" > "$OUTPUT_FILE"

# Compile once (outside the loop)
/usr/local/cuda/bin/nvcc -O3 "$PROGRAM".cu `pkg-config opencv4 --cflags --libs` "$PROGRAM".cpp -o "$PROGRAM"
if [ $? -ne 0 ]; then
    echo "Compilation failed."
    exit 1
fi

# Loop through 1 to 16 cores
for lines in {1..32..5}
do
    echo "===================================================" >> "$OUTPUT_FILE"
    echo "Block lines: $lines " >> "$OUTPUT_FILE"
    echo "===================================================" >> "$OUTPUT_FILE"
    for cols in {1..32..5}
    do
        echo "Block cols: $cols" >> "$OUTPUT_FILE"
        if [ "$MODE" = "anaglyphs" ]; then # ./benchMark_cuda.sh PW2_Ex1-1_cuda statue result 50 anaglyphs true
            ANAGLYPHS=$6
            ./"$PROGRAM" "$IMAGE".jpg "$RESULT".png "$ITER" "$ANAGLYPHS" "$lines" "$cols" >> "$OUTPUT_FILE"
            # xdg-open "$RESULT".png
        fi
        echo "-------------------------------" >> "$OUTPUT_FILE"
    done
done

echo "Benchmark completed. Results saved in $OUTPUT_FILE"











# #!/bin/bash
# # filepath: benchmark_ex2.sh

# PROGRAM=$1
# IMAGE=$2
# MODE=$3
# KSIZE=$4
# SIGMA=$5





# # Create results directory if it doesn't exist
# mkdir -p benchResults


# # Loop through all combinations of TX and TY
# echo "=== Testing Part 1 a 0 ==="
# for i in {1..16}
# do
#     # Run the benchmark with current thread configuration
#     echo "Configuration core=$i" >> "./benchResults/Ex1-1_omp.txt" 
#     g++ "$PROGRAM".cpp -fopenmp `pkg-config opencv4 --cflags` -c
#     g++ "$PROGRAM".o  -fopenmp `pkg-config opencv4 --libs` -lstdc++ -o "$PROGRAM"
#     ./"$PROGRAM" "$IMAGE".jpg "$MODE" "$i" >> "./benchResults/Ex1-1_omp.txt"
    
    

# done

# # # Define input image - replace with your actual input image path
# # INPUT_IMAGE="./statue.jpg"

# # # Fixed parameters
# # ITER=100

# # # Define thread dimensions to test
# # TX_VALUES=(1 2 4 8 16 32)
# # TY_VALUES=(1 2 4 8 16 32)

# # Create results directory if it doesn't exist
# # mkdir -p benchResults
# # # Loop through all combinations of TX and TY
# # echo "Blocks Threads Time TimePerIter IPS" > ./results/ex1cp1_a0.txt
# # echo "=== Testing Part 1 a 0 ==="
# # for tx in "${TX_VALUES[@]}"; do
# #     for ty in "${TY_VALUES[@]}"; do
# #         # Run the benchmark with current thread configuration
# #         ./ex1_cuda_part1.exe "${INPUT_IMAGE}" -i $ITER -a 0 \
# #                  -tx $tx -ty $ty -e >> "./results/ex1cp1_a0.txt"
        
# #         # Append to combined results file
# #         # cat "results/ex2_tx${tx}_ty${ty}.txt" >> results/all_results.txt
        
# #         echo "Configuration TX=$tx, TY=$ty complete"
# #     done
# # done