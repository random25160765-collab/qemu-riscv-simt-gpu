#!/bin/bash
# VPU Fuzzer wrapper — 每轮新进程, 状态隔离
BIN=$(dirname $0)/fuzz_test
N=${1:-100000}
SEED=${2:-0}
E=0; ST=0
for ((i=0; i<N; i++)); do
    OUT=$($BIN -n 1 -q -s $((SEED + i)) 2>&1)
    if [ $? -ne 0 ]; then
        E=$((E+1))
        echo "CRASH round $i seed=$((SEED+i)): $OUT"
        [ $E -ge 10 ] && echo "Too many crashes, stopping" && break
    fi
    ST=$((ST+1))
    [ $((i % 1000)) -eq 0 ] && echo "  $i/$N ok, $E crashes"
done
echo "FUZZ DONE: $ST passed, $E crashes"
[ $E -gt 0 ] && exit 1
exit 0
