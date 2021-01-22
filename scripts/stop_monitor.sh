#!/bin/bash
if [ "$#" -ne 1 ]; then
	exit 1
fi
OUT_DIR=$1
pkill -f mpstat
pkill -f free
pkill -f gpulog
pkill -f nvidia-smi
pkill -f dstat

mv *.log  $OUT_DIR/
mv *.csv  $OUT_DIR/
mv *.out $OUT_DIR/
