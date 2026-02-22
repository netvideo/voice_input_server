"""Qwen3-ASR 推理引擎.

使用 qwen_asr 库的 Qwen3ASRModel 进行语音识别.
"""

import io
import logging
import tempfile
import wave
from pathlib import Path
from typing import Generator, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class QwenASREngine:
    """Qwen3-ASR 推理引擎 - 使用 qwen_asr 库."""
    
    def __init__(
        self,
        model_name: str = "../../Qwen3-ASR-1.7B",
        device: str = "auto",
        dtype: str = "bfloat16",
        max_inference_batch_size: int = 32,
        max_new_tokens: int = 256,
        forced_aligner: Optional[str] = None,
    ):
        """初始化推理引擎.
        
        Args:
            model_name: 模型路径或HuggingFace repo id
            device: 运行设备 (auto/cuda:0/cuda/cpu)
            dtype: 数据类型 (bfloat16/float16/float32)
            max_inference_batch_size: 推理批处理大小，-1表示不分割
            max_new_tokens: 最大生成token数
            forced_aligner: 强制对齐模型路径（可选）
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.dtype = self._get_dtype(dtype)
        self.max_inference_batch_size = max_inference_batch_size
        self.max_new_tokens = max_new_tokens
        self.forced_aligner = forced_aligner
        
        # 加载模型
        self.model = None
        self._load_model()
        
        logger.info(f"Qwen3-ASR引擎初始化完成: {model_name} on {self.device}")
    
    def _get_device(self, device: str) -> str:
        """解析设备字符串."""
        if device == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        return device
    
    def _get_dtype(self, dtype: str):
        """解析数据类型字符串."""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(dtype, torch.bfloat16)
    
    def _load_model(self):
        """加载模型."""
        try:
            from qwen_asr import Qwen3ASRModel
            
            logger.info(f"正在加载模型: {self.model_name}")
            
            # 准备加载参数
            load_kwargs = {
                "dtype": self.dtype,
                "device_map": self.device,
                "max_inference_batch_size": self.max_inference_batch_size,
                "max_new_tokens": self.max_new_tokens,
            }
            
            # 如果有强制对齐器，添加参数
            if self.forced_aligner:
                load_kwargs["forced_aligner"] = self.forced_aligner
            
            # 加载模型
            self.model = Qwen3ASRModel.from_pretrained(
                self.model_name,
                **load_kwargs
            )
            
            logger.info("模型加载完成")
            
        except ImportError:
            logger.error("未安装 qwen_asr 库，请运行: pip install qwen-asr")
            raise
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def preprocess_audio(
        self, 
        audio_data: bytes, 
        sample_rate: int = 16000
    ) -> np.ndarray:
        """预处理音频数据.
        
        Args:
            audio_data: PCM音频数据
            sample_rate: 采样率
            
        Returns:
            预处理后的音频数组 (适合 Qwen3ASRModel.transcribe)
        """
        # 将字节转换为numpy数组
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # 转换为float32并归一化到 [-1, 1]
        audio_array = audio_array.astype(np.float32) / 32768.0
        
        return audio_array
    
    def recognize(
        self, 
        audio_data: bytes,
        language: Optional[str] = None,
        return_timestamps: bool = False,
        context: str = "",
    ) -> dict:
        """识别音频 (使用内存缓冲区).
        
        Args:
            audio_data: PCM音频数据 (16kHz, 16bit, 单声道)
            language: 语言代码 (如 "zh", "en", "zh-CN", "en-US")
            return_timestamps: 是否返回时间戳（需要forced_aligner）
            context: 上下文提示文本
            
        Returns:
            识别结果字典
        """
        try:
            # 使用内存缓冲区而不是文件
            audio_array = self.preprocess_audio(audio_data)
            
            # 转换语言代码格式
            lang = self._normalize_language(language)
            
            # 调用模型识别
            results = self.model.transcribe(
                audio=audio_array,
                context=context,
                language=lang,
                return_time_stamps=return_timestamps,
            )
            
            # 处理结果
            if results and len(results) > 0:
                result = results[0]
                # result 是 ASRTranscription 对象
                # 属性可能包括: text, language, timestamps 等
                text = getattr(result, 'text', str(result))
                
                result_dict = {
                    "text": text,
                    "language": lang or "auto",
                    "confidence": 0.95,  # Qwen3-ASR 可能不直接提供置信度
                }
                
                # 如果有时间戳
                if return_timestamps and hasattr(result, 'timestamps'):
                    result_dict["timestamps"] = result.timestamps
                
                return result_dict
            else:
                return {
                    "text": "",
                    "language": lang or "auto",
                    "confidence": 0.0,
                }
            
        except Exception as e:
            logger.error(f"识别失败: {e}")
            return {"text": "", "error": str(e)}
    
    def _normalize_language(self, language: Optional[str]) -> Optional[str]:
        """标准化语言代码为 Qwen3-ASR 支持的格式.
        
        Qwen3-ASR 支持的语言 (使用英文名称):
        Chinese, English, Cantonese, Arabic, German, French, Spanish,
        Portuguese, Indonesian, Italian, Korean, Russian, Thai, Vietnamese,
        Japanese, Turkish, Hindi, Malay, Dutch, Swedish, Danish, Finnish,
        Polish, Czech, Filipino, Persian, Greek, Romanian, Hungarian, Macedonian
        
        Args:
            language: 原始语言代码或名称
            
        Returns:
            Qwen3-ASR 支持的语言名称，None 表示自动检测
        """
        if not language or language.lower() == "auto":
            return None  # 自动检测时不传language参数
        
        # 语言代码/名称映射到 Qwen3-ASR 格式
        lang_map = {
            # 中文
            "zh": "Chinese",
            "zh-cn": "Chinese",
            "zh-tw": "Chinese",
            "chinese": "Chinese",
            # 英语
            "en": "English",
            "en-us": "English",
            "en-gb": "English",
            "english": "English",
            # 粤语
            "yue": "Cantonese",
            "zh-hk": "Cantonese",
            "cantonese": "Cantonese",
            # 阿拉伯语
            "ar": "Arabic",
            "arabic": "Arabic",
            # 德语
            "de": "German",
            "german": "German",
            # 法语
            "fr": "French",
            "french": "French",
            # 西班牙语
            "es": "Spanish",
            "spanish": "Spanish",
            # 葡萄牙语
            "pt": "Portuguese",
            "portuguese": "Portuguese",
            # 印尼语
            "id": "Indonesian",
            "indonesian": "Indonesian",
            # 意大利语
            "it": "Italian",
            "italian": "Italian",
            # 韩语
            "ko": "Korean",
            "ko-kr": "Korean",
            "korean": "Korean",
            # 俄语
            "ru": "Russian",
            "russian": "Russian",
            # 泰语
            "th": "Thai",
            "thai": "Thai",
            # 越南语
            "vi": "Vietnamese",
            "vietnamese": "Vietnamese",
            # 日语
            "ja": "Japanese",
            "ja-jp": "Japanese",
            "japanese": "Japanese",
            # 土耳其语
            "tr": "Turkish",
            "turkish": "Turkish",
            # 印地语
            "hi": "Hindi",
            "hindi": "Hindi",
            # 马来语
            "ms": "Malay",
            "malay": "Malay",
            # 荷兰语
            "nl": "Dutch",
            "dutch": "Dutch",
            # 瑞典语
            "sv": "Swedish",
            "swedish": "Swedish",
            # 丹麦语
            "da": "Danish",
            "danish": "Danish",
            # 芬兰语
            "fi": "Finnish",
            "finnish": "Finnish",
            # 波兰语
            "pl": "Polish",
            "polish": "Polish",
            # 捷克语
            "cs": "Czech",
            "czech": "Czech",
            # 菲律宾语
            "tl": "Filipino",
            "filipino": "Filipino",
            # 波斯语
            "fa": "Persian",
            "persian": "Persian",
            # 希腊语
            "el": "Greek",
            "greek": "Greek",
            # 罗马尼亚语
            "ro": "Romanian",
            "romanian": "Romanian",
            # 匈牙利语
            "hu": "Hungarian",
            "hungarian": "Hungarian",
            # 马其顿语
            "mk": "Macedonian",
            "macedonian": "Macedonian",
        }
        
        normalized = lang_map.get(language.lower())
        if normalized:
            return normalized
        
        # 如果输入已经是英文名称且在支持列表中，直接返回
        supported_languages = [
            "Chinese", "English", "Cantonese", "Arabic", "German", "French",
            "Spanish", "Portuguese", "Indonesian", "Italian", "Korean", "Russian",
            "Thai", "Vietnamese", "Japanese", "Turkish", "Hindi", "Malay",
            "Dutch", "Swedish", "Danish", "Finnish", "Polish", "Czech",
            "Filipino", "Persian", "Greek", "Romanian", "Hungarian", "Macedonian"
        ]
        
        if language in supported_languages:
            return language
        
        # 默认返回 None (自动检测)
        logger.warning(f"未知的语言代码: {language}，使用自动检测")
        return None
    
    def recognize_stream(
        self, 
        audio_chunks: List[bytes],
        language: Optional[str] = None,
        chunk_length: int = 30,
        context: str = "",
    ) -> Generator[dict, None, None]:
        """流式识别音频.
        
        Args:
            audio_chunks: 音频块列表
            language: 语言代码
            chunk_length: 每个识别块的长度（秒）
            context: 上下文提示文本
            
        Yields:
            识别结果字典
        """
        # 累积音频数据
        buffer = b""
        target_bytes = chunk_length * 16000 * 2  # 16kHz, 16bit = 32 bytes/ms
        
        for chunk in audio_chunks:
            buffer += chunk
            
            # 当缓冲区足够大时进行识别
            if len(buffer) >= target_bytes:
                # 识别当前缓冲区
                result = self.recognize(buffer, language, context=context)
                
                if "error" not in result:
                    yield {
                        "text": result["text"],
                        "is_final": False,
                        "confidence": result.get("confidence", 0.95)
                    }
                
                # 清空缓冲区
                buffer = b""
        
        # 处理剩余音频
        if buffer:
            result = self.recognize(buffer, language, context=context)
            if "error" not in result:
                yield {
                    "text": result["text"],
                    "is_final": True,
                    "confidence": result.get("confidence", 0.95)
                }
    
    def get_model_info(self) -> dict:
        """获取模型信息.
        
        Returns:
            模型信息字典
        """
        # 从模型路径推断参数量
        params = "1.7B"
        if "0.6B" in self.model_name:
            params = "0.6B"
        elif "1.7B" in self.model_name:
            params = "1.7B"
        
        return {
            "model_name": self.model_name,
            "device": self.device,
            "dtype": str(self.dtype).split('.')[-1],
            "max_inference_batch_size": self.max_inference_batch_size,
            "max_new_tokens": self.max_new_tokens,
            "parameters": params,
        }


# 简单的测试
if __name__ == "__main__":
    import sys
    
    print("测试 Qwen3-ASR 引擎")
    print("=" * 50)
    
    # 初始化引擎
    try:
        engine = QwenASREngine(
            model_name="../../Qwen3-ASR-1.7B",
            device="cpu",  # 测试时使用CPU
            dtype="float32",  # CPU使用float32
        )
        
        info = engine.get_model_info()
        print(f"模型: {info['model_name']}")
        print(f"设备: {info['device']}")
        print(f"数据类型: {info['dtype']}")
        print(f"参数量: {info['parameters']}")
        print("✓ 引擎初始化成功")
        
    except Exception as e:
        print(f"✗ 引擎初始化失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
