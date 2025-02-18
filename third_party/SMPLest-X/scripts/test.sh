#!/usr/bin/env bash
TESTSET=$1
CKPT_PATH=$2
CKPT_ID=$3

PYTHONPATH=../:$PYTHONPATH \
python main/test.py \
    --num_gpus 1 \
    --exp_name test_$TESTSET \
    --result_path $CKPT_PATH \
    --ckpt_idx $CKPT_ID \
    --testset $TESTSET
