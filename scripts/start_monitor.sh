#!/bin/bash
SCRIPTS="scripts/" 
mpstat -P ALL 1 > cpu_util.out 2>&1 &  
$SCRIPTS/free.sh & 
dstat -cdnmgyr --output all-utils.csv 2>&1 &  
$SCRIPTS/gpulog.sh &   
