# SGLang Testing - LLM Serving Benchmark Suite

## Project Overview

This is a **local benchmarking toolkit** for evaluating LLM (Large Language Model) inference serving performance. It benchmarks various inference engines including:

- **SGLang** (native and OpenAI-compatible endpoints)
- **vLLM** (OpenAI-compatible endpoints)
- **LMDeploy** (OpenAI-compatible endpoints)
- **TensorRT-LLM** (TRT backend)
- **Truss** and **GServer** (additional backends)

The toolkit measures key performance metrics:
- **Throughput**: Request, input token, and output token throughput
- **Latency**: Time to First Token (TTFT), Inter-Token Latency (ITL), Time per Output Token (TPOT), End-to-End latency
- **Concurrency**: Actual concurrent request handling

## Project Structure

```
/root/sglang-testing/
├── bench_serving.py      # Main benchmark script (primary)
├── bench_serving-1.py    # Alternate benchmark script (similar functionality)
├── bench_serving.py.bak  # Backup of benchmark script
├── bench.sh              # Simple benchmark runner for SGLang
├── batch.sh              # Batch benchmark runner for vLLM
├── new-batch.sh          # Advanced batch runner with priority-based testing
├── pyproject.toml        # Python project configuration (uv package manager)
├── requirements.txt      # Python dependencies
├── uv.lock               # Lock file for uv package manager
├── tokenizer/            # Local tokenizer files
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── logs/                 # Test results and summaries
├── ShareGPT_V3_unfiltered_cleaned_split.json  # ShareGPT dataset for benchmarks
└── *.jsonl               # Benchmark result files (named by backend_date_prompts_input_output.jsonl)
```

## Dependencies

- Python >= 3.9
- aiohttp >= 3.9 (async HTTP client)
- numpy >= 1.23
- requests >= 2.31
- tqdm >= 4.66 (progress bars)
- transformers >= 4.41 (tokenizers)

## Setup

```bash
# Install dependencies using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

## Running Benchmarks

### Basic Usage

```bash
# Run with SGLang backend (native endpoint)
python bench_serving.py --backend sglang --model <model_path> --num-prompts 100

# Run with vLLM backend (OpenAI-compatible)
python bench_serving.py --backend vllm --model <model_path> --host <host> --port 8000

# Run with random dataset (specify input/output lengths)
python bench_serving.py --backend vllm \
    --dataset-name random \
    --model /path/to/model \
    --tokenizer /path/to/tokenizer \
    --random-input-len 16384 \
    --random-output-len 1024 \
    --request-rate 2 \
    --max-concurrency 16 \
    --num-prompts 16
```

### Using Shell Scripts

**bench.sh** - Simple SGLang benchmark:
```bash
./bench.sh
# Configures: INPUT_LEN=3500, OUTPUT_LEN=1500, request_rate=10, concurrency=10
```

**batch.sh** - vLLM batch testing:
```bash
./batch.sh
# Tests multiple configurations with vLLM backend
```

**new-batch.sh** - Advanced batch testing with priority levels:
```bash
# Full test suite
MODEL_PATH=/data00/GLM-4.6-FP8 ./new-batch.sh

# High-priority tests only
MODEL_PATH=/data00/GLM-4.6-FP8 ./new-batch.sh -p

# Maximum input test only
MODEL_PATH=/data00/GLM-4.6-FP8 ./new-batch.sh --max-input-only

# Custom configuration
MODEL_PATH=/data00/model MODEL_NAME=CustomModel ./new-batch.sh -m CustomModel
```

### Key Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--backend` | Inference engine (sglang, vllm, lmdeploy, trt, etc.) | sglang |
| `--dataset-name` | Dataset type (sharegpt, random, generated-shared-prefix) | sharegpt |
| `--model` | Model name or path | (auto-detected) |
| `--tokenizer` | Tokenizer path (defaults to model path) | (same as model) |
| `--num-prompts` | Number of prompts to process | 1000 |
| `--random-input-len` | Input token length (random dataset) | 1024 |
| `--random-output-len` | Output token length (random dataset) | 1024 |
| `--request-rate` | Requests per second (inf = burst) | inf |
| `--max-concurrency` | Maximum concurrent requests | None |
| `--host` | Server host | 0.0.0.0 |
| `--port` | Server port | (backend-specific) |
| `--multi` | Run multiple request rates | False |
| `--request-rate-range` | Rate range (e.g., "2,34,2" or "1,2,4,8") | "2,34,2" |

## Output Files

Results are saved as JSONL files with naming convention:
```
{backend}_{date}_{num_prompts}_{input_len}_{output_len}.jsonl
```

Example: `vllm_1021_16_65536_1024.jsonl`

Each result contains:
- Configuration parameters (backend, dataset, request_rate, etc.)
- Throughput metrics (request/s, tokens/s)
- Latency percentiles (mean, median, p99 for TTFT, TPOT, ITL, E2E)

## Development Conventions

### Code Style
- Python 3.9+ with type hints using dataclasses
- Async/await patterns for concurrent requests
- argparse for CLI argument parsing

### Testing Approach
- Uses ShareGPT dataset or randomly generated prompts
- Supports multiple backends through a unified interface
- Measures streaming performance (TTFT, ITL) via SSE parsing

### Adding New Backends
1. Implement `async_request_*` function following the pattern in `bench_serving.py`
2. Add to `ASYNC_REQUEST_FUNCS` dictionary
3. Update argument parser choices

## Notes

- The benchmark script automatically sets `ulimit` for file descriptors
- Results are appended to JSONL files (not overwritten)
- Logs directory contains detailed test outputs and summaries
- The `tokenizer/` directory contains a local tokenizer for offline testing
