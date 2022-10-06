#!/usr/bin/env bash

# Run this from build directory to keep things simple

if [ $# -lt 2 ]; then
  echo "ERROR: call as"
  echo "$0 dir num_threads"
  exit 1
fi

DIR=$1
NUM_THREADS=$2

FILES=$(ls -d $DIR/*)

TIME=$(date +%s)
OUTFILE="bench-cpu-$TIME.csv"
echo "Writing to $OUTFILE..."
echo "name,#V,l_v,t_v,#E,l_e,t_e,#F,l_f,t_f" > $OUTFILE

for f in $FILES; do
  echo $f
  ./benchmark_cpu "$f" $NUM_THREADS >> $OUTFILE
done

echo "Done writing to $OUTFILE"
