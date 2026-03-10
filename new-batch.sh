#!/bin/bash

# ============================================================================
# 大语言模型性能基准测试脚本 (修正版)
# 修正了原始脚本中的配置错误和逻辑问题
# ============================================================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# ============================================================================
# 1. 环境和路径验证
# ============================================================================

# 模型配置 (可配置变量)
# - MODEL_PATH: 本地模型目录（用于加载 tokenizer / chat_template 检查）
# - MODEL_NAME: OpenAI API 请求里的 model id（可能与路径无关）
MODEL_NAME="${MODEL_NAME:-GLM-4.6-FP8}"  # 可通过环境变量覆盖
MODEL_PATH="${MODEL_PATH:-}"             # 建议显式传入，不要在脚本里拼接

# Backend 配置
BACKEND="${BACKEND:-vllm}"               # 推理引擎: vllm, sglang, lmdeploy 等

# Serving endpoint
BASE_URL="${BASE_URL:-}"                 # e.g. http://10.0.0.174:8000
HOST="${HOST:-10.0.0.174}"
PORT="${PORT:-8000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_SCRIPT="${SCRIPT_DIR}/bench_serving-1.py"
DATASET_PATH="${SCRIPT_DIR}/ShareGPT_V3_unfiltered_cleaned_split.json"

# 测试模式配置
HIGH_PRIORITY_ONLY="${HIGH_PRIORITY_ONLY:-false}"  # 是否仅执行高优先级测试
MAX_INPUT_ONLY="${MAX_INPUT_ONLY:-false}"          # 是否仅测试最大输入组合

# 检查必要的路径和文件
validate_environment() {
    log_info "验证测试环境..."
    log_info "使用模型: ${MODEL_NAME}"
    log_info "模型路径: ${MODEL_PATH}"

    # 检查模型路径
    if [ -z "${MODEL_PATH}" ]; then
        log_error "MODEL_PATH 为空，请通过环境变量传入，例如: MODEL_PATH=/data00/xxx $0"
        exit 1
    fi

    if [ ! -d "${MODEL_PATH}" ]; then
        log_error "模型路径不存在: ${MODEL_PATH}"
        exit 1
    fi

    # bench_serving-1.py 在当前脚本目录
    if [ ! -f "${BENCH_SCRIPT}" ]; then
        log_error "测试脚本不存在: ${BENCH_SCRIPT}"
        exit 1
    fi

    # tokenizer 目录应使用 MODEL_PATH
    if [ ! -d "${MODEL_PATH}" ]; then
        log_error "Tokenizer路径不存在(同MODEL_PATH): ${MODEL_PATH}"
        exit 1
    fi

    # 数据集默认同目录下
    if [ ! -f "${DATASET_PATH}" ]; then
        log_warn "数据集文件可能不存在: ${DATASET_PATH}，继续测试..."
    fi

    log_success "环境验证通过"
}

# ============================================================================
# 2. 测试参数配置 (修正为合理值)
# ============================================================================

# 显示使用说明
show_usage() {
    echo "使用方法:"
    echo "  $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -M, --model-path MODEL_PATH  指定本地模型目录(必须)"
    echo "  -m, --model-name MODEL_NAME  指定请求 payload 里的 model id (默认: GLM-4.6-FP8)"
    echo "  -b, --backend BACKEND        指定推理引擎 (默认: vllm, 可选: sglang, lmdeploy 等)"
    echo "  -u, --base-url BASE_URL      直接指定服务 base url (例如 http://10.0.0.174:8000)"
    echo "      --host HOST              指定 host (默认: 10.0.0.174)"
    echo "      --port PORT              指定 port (默认: 8000)"
    echo "  -p, --high-priority          仅执行高优先级测试"
    echo "  --max-input-only             仅测试最大输入组合"
    echo "  -h, --help                   显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  MODEL_PATH=/data00/GLM-4.6-FP8 $0            # 执行所有测试 (默认 vllm)"
    echo "  $0 -M /data00/GLM-4.6-FP8 -b sglang          # 使用 sglang backend"
    echo "  $0 -M /data00/GLM-4.6-FP8 -p                 # 仅高优先级测试"
    echo "  $0 -M /data00/AnyModelDir -m CustomModelName # model name 与 path 解耦"
    echo "  $0 -M /data00/GLM-4.6-FP8 --max-input-only   # 仅测试最大输入组合"
    echo ""
}

# 解析命令行参数
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -M|--model-path)
                MODEL_PATH="$2"
                shift 2
                ;;
            -m|--model-name)
                MODEL_NAME="$2"
                shift 2
                ;;
            -b|--backend)
                BACKEND="$2"
                shift 2
                ;;
            -u|--base-url)
                BASE_URL="$2"
                shift 2
                ;;
            --host)
                HOST="$2"
                shift 2
                ;;
            --port)
                PORT="$2"
                shift 2
                ;;
            -p|--high-priority)
                HIGH_PRIORITY_ONLY="true"
                shift
                ;;
            --max-input-only)
                MAX_INPUT_ONLY="true"
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                log_error "未知参数: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

# 创建日志目录
mkdir -p logs

# 基于真实使用数据的测试配置
# 数据分析: 输入tokens 8K-68K (平均20-30K)，输出tokens 50-2000 (平均100-300)

# 覆盖所有真实使用场景的测试配置

# 输入长度配置 (覆盖真实数据范围8K-68K的所有关键点)
declare -a input_lens=(8192 16384 32768 49152 65536)  # 8K, 16K, 32K, 48K, 65K: 完整覆盖真实输入范围

# 输出长度配置 (覆盖真实数据50-2081的所有关键场景)
declare -a output_lens=(64 256 512 1024 2048)          # 64, 256, 512, 1024, 2048: 覆盖短中长输出

# 请求速率配置 (基于真实延迟调整，覆盖低中高负载)
declare -a request_rates=(0.5 1 2 4)                    # 0.5, 1, 2, 4 req/s: 从保守到较高负载

# 最大并发数配置 (统一使用16并发)
declare -a max_concurrency=(16)                          # 16 并发: 统一测试标准

# 提示数量配置 (保持合理统计量)
declare -a num_prompts=(16)                              # 16 提示: 平衡统计意义和时间

# 分层测试策略: 优先级分层，总组合数控制在50个以内

# 高优先级组合 (核心场景)
declare -a high_priority_input=(16384 32768 65536)     # 16K, 32K, 65K: 最常见的输入长度
declare -a high_priority_output=(256 512 1024)           # 256, 512, 1024: 常见输出长度
declare -a high_priority_rate=(1 2)                      # 1, 2 req/s: 合理请求率
declare -a high_priority_concurrency=(16)               # 16 并发: 统一并发标准

# 中优先级组合 (边界场景)
declare -a mid_priority_input=(8192 49152)               # 8K, 48K: 边界输入长度
declare -a mid_priority_output=(64 2048)                 # 64, 2048: 边界输出长度
declare -a mid_priority_rate=(0.5 4)                     # 0.5, 4 req/s: 边界请求率
declare -a mid_priority_concurrency=(16)                 # 16 并发: 统一并发标准

# ============================================================================
# 3. 测试执行函数
# ============================================================================

run_single_test() {
    local rate=$1
    local concurrency=$2
    local prompts=$3
    local input_len=$4
    local output_len=$5

    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local log_file="logs/r${rate}_c${concurrency}_p${prompts}_in${input_len}_out${output_len}_${timestamp}.txt"

    log_info "开始测试: 速率=${rate} 并发=${concurrency} 提示=${prompts} 输入=${input_len} 输出=${output_len}"

    # 构建测试命令
    local endpoint_args=""
    if [ -n "${BASE_URL}" ]; then
        endpoint_args="--base-url ${BASE_URL}"
    else
        endpoint_args="--host ${HOST} --port ${PORT}"
    fi

    local cmd="uv run python ${BENCH_SCRIPT} \
        --backend ${BACKEND} \
        --dataset-name random \
        --model ${MODEL_PATH} \
        --model-name ${MODEL_NAME} \
        --dataset-path ${DATASET_PATH} \
        --tokenizer ${MODEL_PATH} \
        --random-input-len ${input_len} \
        --random-output-len ${output_len} \
        --random-range-ratio 1 \
        --request-rate ${rate} \
        --max-concurrency ${concurrency} \
        --num-prompts ${prompts} \
        ${endpoint_args}"

    # 执行测试并记录结果
    echo "========================================" > "${log_file}"
    echo "测试时间: $(date)" >> "${log_file}"
    echo "配置参数:" >> "${log_file}"
    echo "  - 请求速率: ${rate} req/s" >> "${log_file}"
    echo "  - 最大并发: ${concurrency}" >> "${log_file}"
    echo "  - 提示数量: ${prompts}" >> "${log_file}"
    echo "  - 输入长度: ${input_len} tokens" >> "${log_file}"
    echo "  - 输出长度: ${output_len} tokens" >> "${log_file}"
    echo "========================================" >> "${log_file}"
    echo "" >> "${log_file}"

    # 执行测试
    eval "${cmd}" >> "${log_file}" 2>&1
    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        log_success "测试完成: ${log_file}"
        
        # 原始输出测试结果内容
        echo ""
        log_info "=== 测试结果原始输出 ==="
        cat "${log_file}"
        echo "=========================="
        echo ""
    else
        log_error "测试失败: ${log_file} (退出码: ${exit_code})"
    fi

    echo "" >> "${log_file}"
    echo "========================================" >> "${log_file}"
    echo "测试结束: $(date)" >> "${log_file}"
    echo "退出码: ${exit_code}" >> "${log_file}"
    echo "========================================" >> "${log_file}"
}

# ============================================================================
# 4. 分层测试循环 (覆盖所有场景)
# ============================================================================

run_priority_tests() {
    local priority=$1
    local -n inputs_ref=$2
    local -n outputs_ref=$3
    local -n rates_ref=$4
    local -n concurrencies_ref=$5
    local prompts_ref=${6}
    local priority_name=$7
    
    log_info "执行${priority_name}测试..."
    
    local test_count=0
    for rate in "${rates_ref[@]}"; do
        for concurrency in "${concurrencies_ref[@]}"; do
            for input_len in "${inputs_ref[@]}"; do
                for output_len in "${outputs_ref[@]}"; do
                    ((test_count++))
                    run_single_test "$rate" "$concurrency" "$prompts_ref" "$input_len" "$output_len"
                    sleep 1  # 减少延迟
                done
            done
        done
    done
    
    log_success "${priority_name}测试完成! 执行了 ${test_count} 个测试"
    return $test_count
}

run_all_tests() {
    if [ "$MAX_INPUT_ONLY" = "true" ]; then
        log_info "开始执行最大输入组合测试..."
        
        # 仅测试最大输入组合
        local input_len=65536     # 最大输入
        local output_len=1024     # 最大输出
        local rate=2              # 较高负载
        local concurrency=16       # 统一并发
        local prompts=16          # 提示数量
        
        log_info "测试配置: 输入=${input_len} 输出=${output_len} 速率=${rate} 并发=${concurrency} 提示=${prompts}"
        run_single_test "$rate" "$concurrency" "$prompts" "$input_len" "$output_len"
        
        log_success "最大输入组合测试完成! 执行了 1 个测试"
    elif [ "$HIGH_PRIORITY_ONLY" = "true" ]; then
        log_info "开始执行高优先级性能基准测试..."
        
        local total_tests=0
        total_tests=$((${#high_priority_input[@]} * ${#high_priority_output[@]} * ${#high_priority_rate[@]} * ${#high_priority_concurrency[@]}))
        
        log_info "计划执行 ${total_tests} 个高优先级测试组合"
        
        # 仅执行高优先级测试
        local high_tests=0
        run_priority_tests "high" high_priority_input high_priority_output high_priority_rate high_priority_concurrency "${num_prompts[0]}" "高优先级"
        high_tests=$?
        
        log_success "高优先级测试完成! 共执行 ${high_tests} 个测试"
    else
        log_info "开始执行分层性能基准测试..."
        
        local total_tests=0
        local completed_tests=0
        
        # 计算总测试数量
        total_tests=$((${#high_priority_input[@]} * ${#high_priority_output[@]} * ${#high_priority_rate[@]} * ${#high_priority_concurrency[@]}))
        total_tests=$((total_tests + ${#mid_priority_input[@]} * ${#mid_priority_output[@]} * ${#mid_priority_rate[@]} * ${#mid_priority_concurrency[@]}))
        
        log_info "计划执行 ${total_tests} 个测试组合 (高优先级 + 中优先级)"
        
        # 执行高优先级测试
        local high_tests=0
        run_priority_tests "high" high_priority_input high_priority_output high_priority_rate high_priority_concurrency "${num_prompts[0]}" "高优先级"
        high_tests=$?
        completed_tests=$((completed_tests + high_tests))
        log_info "进度: ${completed_tests}/${total_tests}"
        
        # 执行中优先级测试
        local mid_tests=0
        run_priority_tests "mid" mid_priority_input mid_priority_output mid_priority_rate mid_priority_concurrency "${num_prompts[0]}" "中优先级"
        mid_tests=$?
        completed_tests=$((completed_tests + mid_tests))
        
        log_success "所有测试完成! 共执行 ${completed_tests} 个测试 (高优先级: ${high_tests}, 中优先级: ${mid_tests})"
    fi
}

# ============================================================================
# 5. 结果汇总函数
# ============================================================================

summarize_results() {
    log_info "生成测试结果汇总..."

    local summary_file="logs/test_summary_$(date +%Y%m%d_%H%M%S).txt"

    echo "========================================" > "${summary_file}"
    echo "${MODEL_NAME} 性能基准测试结果汇总" >> "${summary_file}"
    echo "测试时间: $(date)" >> "${summary_file}"
    echo "========================================" >> "${summary_file}"
    echo "" >> "${summary_file}"

    # 遍历所有日志文件并提取关键指标
    for log_file in logs/r*_c*_p*_in*_out*_*.txt; do
        if [ -f "$log_file" ]; then
            echo "文件: $log_file" >> "${summary_file}"
            # 提取关键性能指标 (根据实际输出格式调整)
            grep -E "(Request throughput|Input token throughput|Output token throughput|Mean.*Latency)" "$log_file" >> "${summary_file}" 2>/dev/null || echo "  未找到性能指标" >> "${summary_file}"
            echo "" >> "${summary_file}"
        fi
    done

    log_success "结果汇总已保存到: ${summary_file}"
}

# ============================================================================
# 6. 主程序入口
# ============================================================================

main() {
    # 解析命令行参数
    parse_arguments "$@"

    # MODEL_PATH 由外部传入，不在脚本中拼接/推断

    echo "========================================"
    echo "大语言模型性能基准测试脚本 (修正版)"
    echo "测试模型: ${MODEL_NAME}"
    echo "模型路径: ${MODEL_PATH}"
    echo "Backend: ${BACKEND}"
    if [ "$MAX_INPUT_ONLY" = "true" ]; then
        echo "测试模式: 最大输入组合测试"
    elif [ "$HIGH_PRIORITY_ONLY" = "true" ]; then
        echo "测试模式: 仅高优先级测试"
    else
        echo "测试模式: 完整测试 (高优先级 + 中优先级)"
    fi
    echo "测试时间: $(date)"
    echo "========================================"
    echo ""

    # 1. 验证环境
    validate_environment

    # 2. 运行测试
    run_all_tests

    # 3. 汇总结果
    summarize_results

    echo ""
    log_success "测试流程全部完成!"
    echo "请查看 logs/ 目录中的详细测试结果"
}

# 检查是否直接执行此脚本
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
