"""ASR 热词表管理模块.

用于提升语音识别的准确率，通过在识别时提供上下文提示.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("hotword")


class HotwordManager:
    """ASR热词表管理器."""
    
    def __init__(self, config: dict = None):
        """初始化热词管理器.
        
        Args:
            config: 热词配置
                - file: 热词表文件路径
                - enabled: 是否启用
                - context_template: 上下文模板
        """
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.context_template = self.config.get("context_template", "请注意识别以下词语：{hotwords}")
        
        self._hotwords: Dict[str, float] = {}  # word -> weight
        self._hotword_list: List[str] = []
        
        self._load_hotwords()
    
    def _load_hotwords(self):
        """加载热词表."""
        hotword_file = self.config.get("file", "hotword.txt")
        
        if not hotword_file:
            logger.warning("未配置热词表文件")
            return
        
        file_path = Path(hotword_file)
        if not file_path.exists():
            logger.info(f"热词表文件不存在: {hotword_file}")
            return
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    
                    parts = line.split("|")
                    word = parts[0].strip()
                    weight = 1.0
                    if len(parts) > 1:
                        try:
                            weight = float(parts[1].strip())
                            weight = max(0.0, min(1.0, weight))
                        except ValueError:
                            logger.warning(f"无效的权重值: {parts[1]}，使用默认值 1.0")
                            weight = 1.0
                    
                    if word:
                        self._hotwords[word] = weight
            
            self._hotword_list = list(self._hotwords.keys())
            logger.info(f"已加载 {len(self._hotwords)} 个热词")
            
        except Exception as e:
            logger.error(f"加载热词表失败: {e}")
    
    def reload(self):
        """重新加载热词表."""
        logger.info("重新加载热词表...")
        self._hotwords.clear()
        self._hotword_list.clear()
        self._load_hotwords()
    
    def add_hotword(self, word: str, weight: float = 1.0):
        """添加热词.
        
        Args:
            word: 词或短语
            weight: 权重 (0-1)
        """
        if word:
            self._hotwords[word] = weight
            if word not in self._hotword_list:
                self._hotword_list.append(word)
            logger.info(f"添加热词: {word} (权重: {weight})")
    
    def remove_hotword(self, word: str) -> bool:
        """移除热词.
        
        Returns:
            是否成功移除
        """
        if word in self._hotwords:
            del self._hotwords[word]
            self._hotword_list.remove(word)
            logger.info(f"移除热词: {word}")
            return True
        return False
    
    def get_hotwords(self) -> List[dict]:
        """获取所有热词.
        
        Returns:
            热词列表
        """
        return [
            {"word": word, "weight": weight}
            for word, weight in self._hotwords.items()
        ]
    
    def get_context(self) -> str:
        """获取上下文提示字符串.
        
        Returns:
            格式化的上下文提示
        """
        if not self.enabled or not self._hotword_list:
            return ""
        
        hotwords_str = "、".join(self._hotword_list)
        
        if self.context_template:
            return self.context_template.format(hotwords=hotwords_str)
        
        return hotwords_str
    
    def process_result(self, text: str) -> str:
        """后处理识别结果.
        
        可以用于修正识别结果中的热词拼写等.
        
        Args:
            text: 原始识别结果
            
        Returns:
            处理后的文本
        """
        return text
    
    def is_enabled(self) -> bool:
        """是否启用热词."""
        return self.enabled
    
    def set_enabled(self, enabled: bool):
        """设置启用状态."""
        self.enabled = enabled
        logger.info(f"热词表已{'启用' if enabled else '禁用'}")
    
    def get_stats(self) -> dict:
        """获取统计信息."""
        return {
            "enabled": self.enabled,
            "count": len(self._hotwords)
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    test_file = "test_hotword.txt"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("# 测试热词表\n")
        f.write("阿里巴巴\n")
        f.write("opencode|0.8\n")
        f.write("甄嬛传\n")
    
    manager = HotwordManager({
        "file": test_file,
        "enabled": True,
        "context_template": "请注意识别：{hotwords}"
    })
    
    print(f"热词列表: {manager.get_hotwords()}")
    print(f"上下文: {manager.get_context()}")
    
    manager.add_hotword("测试", 0.5)
    print(f"添加后上下文: {manager.get_context()}")
    
    manager.remove_hotword("阿里巴巴")
    print(f"移除后上下文: {manager.get_context()}")
    
    Path(test_file).unlink()
