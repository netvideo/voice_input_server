FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建模型缓存目录
RUN mkdir -p /app/model_cache

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/model_cache

# 暴露端口
EXPOSE 8765

# 启动命令
CMD ["python", "server.py", "--config", "config.yaml"]
