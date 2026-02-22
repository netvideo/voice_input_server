#!/bin/bash

# Qwen3-ASR 服务端启动脚本

set -e

echo "========================================"
echo "   Qwen3-ASR 语音识别服务端"
echo "========================================"
echo ""

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source venv/bin/activate

# 安装依赖
if [ ! -f ".dependencies_installed" ]; then
    echo "安装依赖..."
    pip install -r requirements.txt
    touch .dependencies_installed
fi

# 显示菜单
echo ""
echo "请选择操作:"
echo "  1. 下载模型"
echo "  2. 启动服务端 (GPU/自动)"
echo "  3. 启动服务端 (CPU模式)"
echo "  4. 查看配置"
echo "  5. 退出"
echo ""

read -p "请选择 (1-5): " choice

case $choice in
    1)
        echo ""
        echo "正在下载模型..."
        python download_model.py --model Qwen/Qwen3-ASR-1.7B
        ;;
    2)
        echo ""
        echo "正在启动服务端 (自动检测设备)..."
        python server.py --config config.yaml
        ;;
    3)
        echo ""
        echo "正在启动服务端 (CPU模式)..."
        python server.py --device cpu --config config.yaml
        ;;
    4)
        echo ""
        echo "当前配置:"
        echo "----------------------------------------"
        cat config.yaml
        echo "----------------------------------------"
        read -p "按回车键继续..."
        ;;
    5)
        echo "退出"
        exit 0
        ;;
    *)
        echo "无效选择"
        ;;
esac
