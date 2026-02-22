# Qwen3-ASR 语音识别服务端

基于 [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) 模型的实时语音识别服务端，使用 `qwen_asr` 库的 `Qwen3ASRModel`。

## 模型特性

- **模型**: Qwen3-ASR-1.7B (也支持 Qwen3-ASR-0.6B)
- **支持语言**: 30 种语言及方言
- **参数**: 1.7B 参数
- **性能**: 开源模型 SOTA 水平
- **编码器**: AuT 语音编码器
- **许可证**: Apache-2.0（可商用）
- **热词支持**: 可配置热词表提升识别准确率

## 系统要求

### 硬件要求

| 配置 | 显存 | 适用场景 |
|------|------|----------|
| **最低配置** | 4GB | 0.6B模型，低并发 |
| **推荐配置** | 8GB+ | 1.7B模型，单并发 |
| **高性能** | 16GB+ | 1.7B模型，多并发 |

### 软件要求

- Python 3.9+
- CUDA 11.8+ (GPU推理) 或 CPU
- 8GB+ RAM

## 快速开始

### 1. 安装依赖

```bash
# 克隆仓库
cd voice_input_server

# 创建虚拟环境
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

**依赖说明:**
- `qwen-asr>=0.0.6` - Qwen3-ASR 模型库
- `torch>=2.0.0` - PyTorch 深度学习框架
- `transformers>=4.36.0` - HuggingFace Transformers
- `websockets>=12.0` - WebSocket 服务端

### 2. 下载模型

```bash
# 方式1: 自动下载（首次运行时会从 HuggingFace 下载）
python download_model.py

# 方式2: 手动下载到本地
# 从 https://huggingface.co/Qwen/Qwen3-ASR-1.7B 下载模型
# 解压到项目目录，如 ./Qwen3-ASR-1.7B

# 方式3: 使用镜像源（国内用户）
export HF_ENDPOINT=https://hf-mirror.com
python download_model.py
```

### 3. 配置服务端

编辑 `config.yaml`:

```yaml
# 模型配置
model:
  # 模型路径或 HuggingFace repo id
  # 本地路径: "Qwen3-ASR-1.7B" (相对于当前目录)
  # HuggingFace: "Qwen/Qwen3-ASR-1.7B"
  name: "Qwen3-ASR-1.7B"
  
  # 运行设备: auto, cuda:0, cuda, cpu
  device: "auto"
  
  # 数据类型: bfloat16 (推荐), float16, float32
  # bfloat16: 推荐用于 RTX 30/40系列, A100, H100
  # float16: 用于较旧的 GPU
  # float32: 用于 CPU
  dtype: "bfloat16"
  
  # 推理批处理大小，-1 表示不进行分割
  max_inference_batch_size: 32
  
  # 最大生成 token 数
  max_new_tokens: 256
  
  # 强制对齐模型路径（可选）
  # forced_aligner: "./Qwen3-ASR-ForcedAligner"
  
# 服务器配置
server:
  host: "0.0.0.0"
  port: 8765
  max_connections: 5
  timeout: 300
  
# 音频配置
audio:
  sample_rate: 16000
  chunk_duration: 0.1  # 100ms
  max_audio_length: 60  # 最大音频长度（秒）
  
# 识别配置
recognition:
  language: "auto"  # 默认语言: auto(自动检测) 或具体语言名称
  return_timestamps: false
  context: ""
  
# 性能配置
performance:
  use_fp16: true
  compile_model: false  # PyTorch 2.0+ 编译优化
  num_threads: 4  # CPU线程数
  stream_mode: true  # 流式识别模式

# 热词表配置 (用于提升识别准确率)
hotword:
  # 热词表文件路径 (每行一个词或短语)
  file: hotword.txt
  # 是否启用热词表
  enabled: true
  # 上下文模板: {hotwords} 会被替换为热词列表
  context_template: "请注意识别以下词语：{hotwords}"
```

### 4. 启动服务端

```bash
# 启动服务
python server.py

# 或者使用配置文件
python server.py --config config.yaml

# GPU模式（自动检测）
python server.py --device cuda

# CPU模式
python server.py --device cpu

# 指定模型
python server.py --model Qwen3-ASR-0.6B
```

### 5. 测试连接

```bash
# 使用客户端测试
cd ../voice_input
python voice_input_app.py
```

## 项目结构

```
voice_input_server/
├── server.py              # 服务端主程序
├── asr_engine.py          # Qwen3-ASR 推理引擎
├── hotword.py             # 热词表管理模块
├── config.yaml            # 配置文件
├── hotword.txt            # 热词表文件
├── requirements.txt       # 依赖列表
├── download_model.py      # 模型下载脚本
└── README.md              # 本文档
```

## 热词表功能

服务端支持热词表功能，可提升特定词语的识别准确率。

### 配置热词表

在 `config.yaml` 中启用热词表：

```yaml
hotword:
  file: hotword.txt
  enabled: true
  context_template: "请注意识别以下词语：{hotwords}"
```

### 编辑热词表

编辑 `hotword.txt` 文件，每行一个词：

```txt
# 格式: 词语 或 词语|权重(0-1)
# 权重越高，识别优先级越高

# 公司名
阿里巴巴|0.9
腾讯|0.9

# 技术术语
Python|0.9
JavaScript|0.9
人工智能|0.9
```

### 热词管理 API

客户端可以通过 WebSocket 动态管理热词：

```json
// 添加热词
{"type": "hotword", "action": "add", "word": "词语", "weight": 1.0}

// 移除热词
{"type": "hotword", "action": "remove", "word": "词语"}

// 列出热词
{"type": "hotword", "action": "list"}

// 重新加载热词表
{"type": "hotword", "action": "reload"}

// 启用/禁用热词
{"type": "hotword", "action": "enable", "enabled": false}
```

### 热词工作原理

热词会在识别时作为上下文提示传递给 ASR 模型：
- 上下文模板中的 `{hotwords}` 会被替换为热词列表
- 模型会优先识别这些词语
- 适用于人名、专业术语、品牌名等

## 语言支持

Qwen3-ASR 支持 30 种语言：

| 语言 | 英文名称 | ISO 代码 |
|------|---------|---------|
| 中文 | Chinese | zh |
| 英语 | English | en |
| 粤语 | Cantonese | yue |
| 日语 | Japanese | ja |
| 韩语 | Korean | ko |
| 阿拉伯语 | Arabic | ar |
| 德语 | German | de |
| 法语 | French | fr |
| 西班牙语 | Spanish | es |
| 葡萄牙语 | Portuguese | pt |
| 印尼语 | Indonesian | id |
| 意大利语 | Italian | it |
| 俄语 | Russian | ru |
| 泰语 | Thai | th |
| 越南语 | Vietnamese | vi |
| 土耳其语 | Turkish | tr |
| 印地语 | Hindi | hi |
| 马来语 | Malay | ms |
| 荷兰语 | Dutch | nl |
| 瑞典语 | Swedish | sv |
| 丹麦语 | Danish | da |
| 芬兰语 | Finnish | fi |
| 波兰语 | Polish | pl |
| 捷克语 | Czech | cs |
| 菲律宾语 | Filipino | tl |
| 波斯语 | Persian | fa |
| 希腊语 | Greek | el |
| 罗马尼亚语 | Romanian | ro |
| 匈牙利语 | Hungarian | hu |
| 马其顿语 | Macedonian | mk |

**使用 `auto` 启用自动语言检测。**

## 性能优化

### GPU 优化

```yaml
# config.yaml
model:
  dtype: "bfloat16"  # 推荐用于支持的 GPU
  
performance:
  use_fp16: true
  compile_model: false  # PyTorch 2.0+ 可尝试开启
```

**推荐 GPU:**
- RTX 30/40 系列（支持 bfloat16）
- A100, H100（支持 bfloat16）
- 较旧的 GPU 使用 float16

### CPU 优化

```yaml
# config.yaml
model:
  dtype: "float32"  # CPU 必须使用 float32
  device: "cpu"
  
performance:
  num_threads: 4  # 根据 CPU 核心数调整
```

### 显存不足解决方案

```bash
# 使用 0.6B 模型（显存需求更低）
python server.py --model Qwen/Qwen3-ASR-0.6B

# 或者减小批处理大小
# 在 config.yaml 中设置:
# max_inference_batch_size: 8
```

## API 协议

服务端遵循 ASR WebSocket 协议：

### 连接

```
ws://host:port
```

### 配置消息

```json
{
  "type": "config",
  "data": {
    "sample_rate": 16000,
    "language": "auto",
    "enable_punctuation": true,
    "enable_itn": true
  }
}
```

**语言设置:**
- `auto` - 自动检测语言
- `Chinese`, `English`, `Japanese`, `Korean` 等 - 指定语言

### 音频消息

```json
{
  "type": "audio",
  "seq": 1,
  "data": "base64_encoded_pcm_audio"
}
```

### 识别结果

```json
{
  "type": "result",
  "code": 0,
  "message": "success",
  "data": {
    "utterance_id": "session_id",
    "is_final": true,
    "text": "识别文本",
    "confidence": 0.95,
    "start_time_ms": 0,
    "end_time_ms": 5000
  }
}
```

### 结束消息

```json
{
  "type": "end",
  "seq": 100
}
```

## Docker 部署

### 构建镜像

```bash
docker build -t voice-asr-server .
```

### 运行容器

```bash
docker run -d \
  --name voice-asr \
  --gpus all \
  -p 8765:8765 \
  -v $(pwd)/model_cache:/app/model_cache \
  voice-asr-server
```

## 故障排除

### 模型加载失败

**错误**: `Repo id must be in the form 'repo_name' or 'namespace/repo_name'`

**解决**: 确保模型路径正确，本地路径不要加 `./`：
```yaml
model:
  name: "Qwen3-ASR-1.7B"  # 正确
  # name: "./Qwen3-ASR-1.7B"  # 错误！
```

### 显存不足 (OOM)

```bash
# 使用更小的模型
python server.py --model Qwen/Qwen3-ASR-0.6B

# 或者减小批处理大小
# 在 config.yaml 中:
# max_inference_batch_size: 8
```

### 模型下载慢

```bash
# 使用镜像源
export HF_ENDPOINT=https://hf-mirror.com
python download_model.py

# 或者手动下载
# 从 https://hf-mirror.com/Qwen/Qwen3-ASR-1.7B 下载
```

### 首次启动慢

首次启动需要下载/加载模型（约 3-4GB），请耐心等待。后续启动会更快。

### bfloat16 不支持

**错误**: `RuntimeError: bfloat16 is not supported on this GPU`

**解决**: 改用 float16 或 float32：
```yaml
model:
  dtype: "float16"  # 或 "float32"
```

## 监控与日志

服务端会自动记录日志到 `server.log`：

```bash
# 查看实时日志
tail -f server.log

# 查看错误日志
grep ERROR server.log
```

## 许可证

本项目遵循 Apache-2.0 许可证。
Qwen3-ASR 模型同样遵循 Apache-2.0 许可证，可自由商用。

## 参考链接

- [Qwen3-ASR GitHub](https://github.com/QwenLM/Qwen3-ASR)
- [Qwen3-ASR HuggingFace](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- [Qwen 官方文档](https://qwen.readthedocs.io/)
