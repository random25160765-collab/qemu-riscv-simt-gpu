#!/bin/bash
# fuzz-run: 持续 fuzz, 记录 crash seed + 指令 dump
# Usage: fuzz-run [rounds] [seed_start]
FUZZ_DIR=$(cd $(dirname $0)/.. && pwd)
BIN=$FUZZ_DIR/fuzz_test
N=${1:-100000}
SEED_START=${2:-0}
LOG=$FUZZ_DIR/crashes.log
DUMP_DIR=$FUZZ_DIR/crashes

mkdir -p $DUMP_DIR

echo "=== VPU Fuzzer Run ==="
echo "Rounds: $N"
echo "Seed start: $SEED_START"
echo "Log: $LOG"
echo "Crash dumps: $DUMP_DIR/"
echo

PASS=0
CRASH=0
for ((i=0; i<N; i++)); do
    SEED=$((SEED_START + i))
    OUT=$($BIN -n 1 -s $SEED -d 2>&1)
    RC=$?
    if [ $RC -ne 0 ]; then
        CRASH=$((CRASH+1))
        echo "$OUT" > $DUMP_DIR/seed_0x$(printf '%x' $SEED).txt
        echo "[$i] CRASH seed=0x$(printf '%x' $SEED) exit=$RC" | tee -a $LOG
        # 只保留前 100 个 crash dump
        if [ $(ls $DUMP_DIR | wc -l) -gt 100 ]; then
            ls -t $DUMP_DIR | tail -n +101 | xargs -I{} rm $DUMP_DIR/{}
        fi
    else
        PASS=$((PASS+1))
    fi
    if [ $((i % 500)) -eq 0 ] && [ $i -gt 0 ]; then
        echo "  [$i/$N] pass=$PASS crash=$CRASH ($(echo "scale=1; $CRASH*100/$i" | bc)%)"
    fi
done

echo
echo "=== Done ==="
echo "Passed: $PASS"
echo "Crashes: $CRASH ($(echo "scale=1; $CRASH*100/$N" | bc)%)"
echo "Crash seeds:"
ls $DUMP_DIR/ 2>/dev/null | head -10
