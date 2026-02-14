#!/bin/bash
set -ex
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo 'SCRIPT_DIR: ' $SCRIPT_DIR

# 定义 input 和 output 长度
INPUT_LEN=3500
OUTPUT_LEN=1500

declare -a request_rates=(10)
declare -a max_concurrency=(10)
declare -a num_prompts=(100)

DATA_JSON_PATH=$SCRIPT_DIR/ShareGPT_V3_unfiltered_cleaned_split.json
if [ ! -f $DATA_JSON_PATH ]; then
    wget https://vemlp-demo-models.tos-cn-beijing.volces.com/vemlp-third-party-warehouse/ShareGPT_V3_unfiltered_cleaned_split.json
    mv ShareGPT_V3_unfiltered_cleaned_split.json $SCRIPT_DIR
fi

i=0
RATE=${request_rates[$i]}
CONCURRENCY=${max_concurrency[$i]}
PROMPTS=${num_prompts[$i]}
python3 $SCRIPT_DIR/bench_serving.py --backend sglang-oai \
        --dataset-name random \
        --model ds \
        --tokenizer $SCRIPT_DIR/tokenizer/ \
        --dataset-path $DATA_JSON_PATH \
        --random-input-len $INPUT_LEN \
        --random-output-len $OUTPUT_LEN \
        --random-range-ratio 1 \
        --request-rate $RATE \
        --max-concurrency $CONCURRENCY \
        --num-prompts $PROMPTS \
        --host 0.0.0.0 --port 80
