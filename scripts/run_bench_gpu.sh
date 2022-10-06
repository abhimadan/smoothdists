#!/usr/bin/env bash

# Run this from build directory to keep things simple

if [ $# -lt 1 ]; then
  echo "ERROR: call as"
  echo "$0 dir"
  exit 1
fi

DIR=$1

FILES=$(ls -d $DIR/*)

TIME=$(date +%s)
OUTFILE="bench-gpu-$TIME.csv"
echo "Writing to $OUTFILE..."
echo "name,#V,t_v,#E,t_e,#F,t_f" > $OUTFILE

for f in $FILES; do
  echo $f
  ./benchmark_gpu "$f" >> $OUTFILE
done

echo "Done writing to $OUTFILE"
