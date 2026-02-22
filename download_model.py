"""模型下载脚本.

自动下载 Qwen3-ASR 模型到本地缓存.
"""

import argparse
import os
import sys
from pathlib import Path

def download_model(model_name: str, cache_dir: str = None):
    """下载模型.
    
    Args:
        model_name: 模型名称
        cache_dir: 缓存目录
    """
    print(f"正在下载模型: {model_name}")
    print("这可能需要几分钟到几十分钟，取决于网络速度...")
    print()
    
    try:
        # 设置缓存目录
        if cache_dir:
            os.environ["HF_HOME"] = cache_dir
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        print("正在使用 qwen_asr 库下载模型...")
        print("模型大小约 3-4GB，请耐心等待...")
        print()
        
        from qwen_asr import Qwen3ASRModel
        
        model = Qwen3ASRModel.from_pretrained(
            model_name,
            dtype="float32",
            device_map="cpu",
        )
        
        print("✓ 模型下载完成")
        print()
        
        print("下载成功！现在可以启动服务端了。")
        print(f"运行: python server.py --model {model_name}")
        
        return True
        
    except ImportError:
        print("\n✗ 未安装 qwen_asr 库")
        print()
        print("请先安装依赖:")
        print("  pip install qwen-asr")
        return False
        
    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        print()
        print("可能的解决方案:")
        print("1. 检查网络连接")
        print("2. 设置镜像源: export HF_ENDPOINT=https://hf-mirror.com")
        print("3. 使用代理")
        print("4. 手动下载模型并指定本地路径")
        return False


def main():
    """主函数."""
    parser = argparse.ArgumentParser(
        description="下载 Qwen3-ASR 模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 下载 1.7B 模型
  python download_model.py
  
  # 下载 0.6B 模型
  python download_model.py --model Qwen/Qwen3-ASR-0.6B
  
  # 指定缓存目录
  python download_model.py --cache-dir ./model_cache
        """
    )
    
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-ASR-1.7B",
        choices=["Qwen/Qwen3-ASR-1.7B", "Qwen/Qwen3-ASR-0.6B"],
        help="要下载的模型 (默认: Qwen/Qwen3-ASR-1.7B)"
    )
    parser.add_argument(
        "--cache-dir",
        help="模型缓存目录"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Qwen3-ASR 模型下载工具")
    print("=" * 60)
    print()
    
    success = download_model(args.model, args.cache_dir)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
