#!/usr/bin/env bash
SEQ=$1

PYTHONPATH=../:./third_party/SMPLest-X:./third_party/DPVO:$PYTHONPATH \
python whac/inference.py \
    --seq_name $SEQ \
    --visualize