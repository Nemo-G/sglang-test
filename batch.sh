#!/bin/bash
# 定义 input 和 output 长度
declare -a input_lens=(81920) # 3500)
declare -a output_lens=(8192) # 1500)
# 创建 logs 目录（如果不存在）
mkdir -p logs
# 定义 request-rate 和 max-concurrency
declare -a request_rates=(32) #(1 4 8 16 32 48 64)
declare -a max_concurrency=(8) #(1 4 8 16 32 48 64)
declare -a num_prompts=(16) #(4 16 32 64 128 192 256)
# 运行 benchmark 测试
for i in "${!request_rates[@]}"; do
    RATE=${request_rates[$i]}
    CONCURRENCY=${max_concurrency[$i]}
    PROMPTS=${num_prompts[$i]}
    echo "Running benchmark with request rate $RATE, concurrency $CONCURRENCY, prompts $PROMPTS"
    for j in "${!input_lens[@]}"; do
        INPUT_LEN=${input_lens[$j]}
        OUTPUT_LEN=${output_lens[$j]}
        echo "INPUT LEN: $INPUT_LEN, OUTPUT LEN: $OUTPUT_LEN"
        python3 /root/vllm/batchtest/xllm/sglang-testing/bench_serving-1.py  --backend vllm \
            --dataset-name random \
	    --model  /data00/GLM-4.6-FP8 \
            --dataset-path /root/vllm/batchtest/xllm/sglang-testing/ShareGPT_V3_unfiltered_cleaned_split.json \
            --tokenizer ./tokenizer/ \
            --random-input-len $INPUT_LEN \
            --random-output-len $OUTPUT_LEN \
            --random-range-ratio 1 \
            --request-rate $RATE \
            --max-concurrency $CONCURRENCY \
            --num-prompts $PROMPTS \
            --host 10.0.0.174 --port 8000
        # > logs/${CONCURRENCY}_fp8_r1_${INPUT_LEN}_${OUTPUT_LEN}.txt
        wait
    done
done
