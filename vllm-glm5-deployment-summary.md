# vLLM GLM-5-FP8 部署总结

## 1. 初始问题

- **模型兼容性错误**：GLM-5-FP8 模型使用 `glm_moe_dsa` 架构，vLLM 默认 transformers 版本不支持
- **CUDA 内核错误**：`no kernel image is available for execution on the device` - 基础镜像的 CUDA 版本与 GPU compute capability (10.3) 不匹配

## 2. 解决方案

### 2.1 自定义镜像构建

**Dockerfile** (`/data/Dockerfile.vllm-glm5`):
```dockerfile
FROM docker.1ms.run/vllm/vllm-openai:cu130-nightly-x86_64

ARG http_proxy=http://212.50.249.225:1080
ARG https_proxy=http://212.50.249.225:1080

ENV http_proxy=${http_proxy} \
    https_proxy=${https_proxy}

# 安装 git（用于安装 transformers）
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 安装最新版 transformers（支持 glm_moe_dsa）
RUN uv pip install --system git+https://github.com/huggingface/transformers.git
```

**本地镜像标签**:
- `vllm-glm5:cu130-nightly-transformers-git`
- `docker.1ms.run/vllm/vllm-openai:glm5-cu130-nightly-transformers-git` (仅本地 tag，未 push)

### 2.2 基础镜像选择

- ❌ `vllm/vllm-openai:v0.17.1-x86_64` - CUDA kernel 不兼容
- ✅ `vllm/vllm-openai:cu130-nightly-x86_64` - 支持 compute capability 10.3 (NVIDIA B300)

## 3. Speculative Decoding 问题

### 3.1 问题现象

使用 `--speculative-config.method mtp --speculative-config.num_speculative_tokens 2` 时：
- 工具调用参数在 stream 输出中出现重复
- 参数被错误修改（如 `{"a":2,"b":3}` 变成 `{"a":2,"b":5}`）
- 模型反复调用工具，无法正确完成任务

### 3.2 根本原因

MTP (Multi-Token Prediction) speculative decoding 会：
1. 使用草稿模型预测多个 token
2. 主模型接受/拒绝这些 token
3. 在结构化输出（tool_calls JSON）的增量拼接过程中， speculative 接受/拒绝会导致：
   - 重复片段
   - 缺失引号/逗号
   - 参数被"修正"成错误值

### 3.3 解决方案

- **方案 A (已采用)**: 完全移除 MTP speculative decoding
- **方案 B (备选)**: 保留 MTP 但将 `num_speculative_tokens` 降为 1（提升稳定性但仍有风险）

## 4. DeepGemm 缓存配置

### 4.1 初始配置（错误）

```yaml
environment:
  VLLM_DEEPGEMM_CACHE_DIR: "/root/.cache/deepgemm"  # ❌ vLLM 不识别此变量
```

**日志警告**:
```
WARNING Unknown vLLM environment variable detected: VLLM_DEEPGEMM_CACHE_DIR
```

### 4.2 正确配置

```yaml
volumes:
  - /data/sglang-cache:/root/.cache/deepgemm

environment:
  VLLM_CACHE_ROOT: "/root/.cache/deepgemm"          # ✅ vLLM 支持的缓存根目录
  FLASHINFER_CACHE_DIR: "/root/.cache/deepgemm"      # ✅ FlashInfer 缓存目录
```

## 5. 最终部署配置

### 5.1 docker-compose.yaml

```yaml
vllm-glm5:
  image: vllm-glm5:cu130-nightly-transformers-git
  container_name: vllm-glm5
  network_mode: "host"
  ipc: "host"
  shm_size: "32g"
  devices:
    - nvidia.com/gpu=all
  volumes:
    - /data:/data
    - /data/sglang-cache:/root/.cache/deepgemm
  environment:
    NCCL_IB_DISABLE: "1"
    NCCL_DEBUG: "INFO"
    VLLM_CACHE_ROOT: "/root/.cache/deepgemm"
    FLASHINFER_CACHE_DIR: "/root/.cache/deepgemm"
  command: >
    /data/GLM-5-FP8
    --tensor-parallel-size 8
    --tool-call-parser glm47
    --reasoning-parser glm45
    --enable-auto-tool-choice
    --served-model-name glm-5-fp8
```

### 5.2 启动参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `--tensor-parallel-size` | 8 | 张量并行度（8 GPU） |
| `--tool-call-parser` | glm47 | GLM-4.7 工具调用解析器 |
| `--reasoning-parser` | glm45 | GLM-4.5 推理解析器 |
| `--enable-auto-tool-choice` | - | 启用自动工具选择 |
| `--served-model-name` | glm-5-fp8 | 服务端模型名称 |

**已移除参数**:
- `--speculative-config.method mtp` - 移除以提升工具调用稳定性
- `--speculative-config.num_speculative_tokens 1` - 移除以避免参数错误

## 6. 验证结果

### 6.1 模型加载

```
Resolved architecture: GlmMoeDsaForCausalLM
Using max model len 202752
```

### 6.2 工具调用测试

成功测试场景：
- ✅ 非流式工具调用 (`stream=false`)
- ✅ 流式工具调用 (`stream=true`)
- ✅ 自动工具选择 (`tool_choice=auto`)
- ✅ 禁用工具调用 (`tool_choice=none`)

### 6.3 端口监听

- vLLM: `http://127.0.0.1:8000` (OpenAI 兼容 API)
- SGLang: `http://127.0.0.1:30000` (对比测试用)

## 7. 关键发现

1. **模型架构支持**：GLM-5-FP8 需要最新版 transformers (git master)
2. **CUDA 兼容性**：必须使用与 GPU compute capability 匹配的基础镜像
3. **Speculative Decoding**：对结构化输出（tool_calls JSON）有显著影响，建议在生产环境中禁用或谨慎使用
4. **缓存配置**：使用 `VLLM_CACHE_ROOT` 而非 `VLLM_DEEPGEMM_CACHE_DIR`
5. **工具调用稳定性**：移除 MTP 后，工具参数输出更加稳定可靠

## 8. 后续建议

### 8.1 如果需要 speculative decoding

- 仅在非工具调用场景使用
- 或将 `num_speculative_tokens` 限制为 1
- 严格测试工具调用场景的稳定性

### 8.2 监控指标

关注以下指标以确保服务稳定：
- tool_calls 的参数完整性
- 流式输出的 finish_reason 分布
- SpecDecoding metrics（如果启用）
- Prefix cache hit rate

### 8.3 客户端适配

- 确保 SSE 解析器正确处理 `data: [DONE]` 标记
- 正确聚合 `tool_calls[].function.arguments`（增量拼接）
- 处理 `reasoning` 字段（GLM 特有）

## 9. 文件清单

- `/data/Dockerfile.vllm-glm5` - 自定义镜像构建文件
- `/data/docker-compose.yaml` - 服务编排配置
- `/data/vllm-glm5-image/` - Docker 构建上下文
- `/data/sglang-cache/` - DeepGemm 缓存持久化目录

---

**生成时间**: 2026-03-13
**vLLM 版本**: 0.17.1rc1.dev88+g36735fd77
**Transformers 版本**: 5.3.0.dev0 (from git)
**GPU**: NVIDIA B300 SXM6 AC (compute capability 10.3)