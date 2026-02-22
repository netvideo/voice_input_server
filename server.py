"""Qwen3-ASR 服务端主程序.

基于 WebSocket 的实时语音识别服务端.
"""

import argparse
import asyncio
import base64
import json
import logging
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml
import websockets
from websockets.exceptions import ConnectionClosed

from asr_engine import QwenASREngine
from hotword import HotwordManager
from itn import itn


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('server.log')
    ]
)
logger = logging.getLogger(__name__)


class ASRSession:
    """ASR会话管理."""
    
    def __init__(self, session_id: str, engine: QwenASREngine, config: dict, hotword_manager: Optional[HotwordManager] = None):
        """初始化会话.
        
        Args:
            session_id: 会话ID
            engine: ASR引擎
            config: 配置参数
            hotword_manager: 热词管理器
        """
        self.session_id = session_id
        self.engine = engine
        self.config = config
        self.hotword_manager = hotword_manager
        
        # 音频缓冲区（用于增量识别）
        self.audio_buffer: List[bytes] = []
        self.buffer_size = 0
        
        # 完整音频数据（用于最终识别）
        self.total_audio: List[bytes] = []
        
        # 识别状态
        self.is_recognizing = False
        self.language = config.get("language", "auto")
        self.enable_itn = config.get("enable_itn", True)
        
        # 统计
        self.start_time = time.time()
        self.total_audio_bytes = 0
    
    def _apply_itn(self, text: str) -> str:
        """应用ITN转换."""
        if self.enable_itn and text:
            return itn(text, self.language)
        return text
    
    def add_audio(self, audio_data: bytes):
        """添加音频数据.
        
        Args:
            audio_data: PCM音频数据
        """
        self.audio_buffer.append(audio_data)
        self.total_audio.append(audio_data)
        self.buffer_size += len(audio_data)
        self.total_audio_bytes += len(audio_data)
    
    def recognize(self, is_final: bool = False) -> Optional[dict]:
        """执行识别.
        
        Args:
            is_final: 是否为最终识别
            
        Returns:
            识别结果或None
        """
        # 获取热词上下文
        context = ""
        if self.hotword_manager and self.hotword_manager.is_enabled():
            context = self.hotword_manager.get_context()
        
        if is_final:
            # 最终识别：使用所有音频数据
            if not self.total_audio:
                return None
            audio_data = b"".join(self.total_audio)
            # 清空所有缓冲区
            self.audio_buffer = []
            self.total_audio = []
            self.buffer_size = 0
        else:
            # 中间识别：使用当前缓冲区
            if not self.audio_buffer:
                return None
            audio_data = b"".join(self.audio_buffer)
            # 只清空当前缓冲区，保留完整音频
            self.audio_buffer = []
            self.buffer_size = 0
        
        # 执行识别（传入上下文）
        result = self.engine.recognize(audio_data, language=self.language, context=context)
        
        return result
    
    def get_stats(self) -> dict:
        """获取会话统计.
        
        Returns:
            统计信息
        """
        duration = time.time() - self.start_time
        return {
            "session_id": self.session_id,
            "duration": duration,
            "total_audio_bytes": self.total_audio_bytes,
            "audio_duration": self.total_audio_bytes / 32000  # 16kHz 16bit
        }


class ASRServer:
    """ASR WebSocket服务器."""
    
    def __init__(self, config: dict):
        """初始化服务器.
        
        Args:
            config: 配置字典
        """
        self.config = config
        
        # 服务器配置
        server_cfg = config.get("server", {})
        self.host = server_cfg.get("host", "0.0.0.0")
        self.port = server_cfg.get("port", 8080)
        self.max_connections = server_cfg.get("max_connections", 10)
        
        # 线程池 - 用于执行识别任务，避免阻塞事件循环
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="asr_worker")
        
        # 初始化ASR引擎
        model_cfg = config.get("model", {})
        self.engine = QwenASREngine(
            model_name=model_cfg.get("name", "Qwen3-ASR-1.7B"),
            device=model_cfg.get("device", "auto"),
            dtype=model_cfg.get("dtype", "bfloat16"),
            max_inference_batch_size=model_cfg.get("max_inference_batch_size", 32),
            max_new_tokens=model_cfg.get("max_new_tokens", 256),
            forced_aligner=model_cfg.get("forced_aligner", None),
        )
        
        # 初始化热词管理器
        hotword_cfg = config.get("hotword", {})
        self.hotword_manager = HotwordManager(hotword_cfg)
        
        # 会话管理
        self.sessions: Dict[str, ASRSession] = {}
        self.connections: Set = set()
        
        # 统计
        self.stats = {
            "total_connections": 0,
            "active_connections": 0,
            "total_sessions": 0,
            "total_recognitions": 0
        }
        
        # 运行标志
        self._running = False
    
    async def handle_client(self, websocket):
        """处理客户端连接.
        
        Args:
            websocket: WebSocket连接
        """
        client_id = f"client_{id(websocket)}"
        session_id = f"session_{int(time.time() * 1000)}"
        
        # 检查连接数限制
        if len(self.connections) >= self.max_connections:
            logger.warning(f"连接数超限，拒绝连接: {client_id}")
            await websocket.close(code=1013, reason="Server overloaded")
            return
        
        self.connections.add(websocket)
        self.stats["total_connections"] += 1
        self.stats["active_connections"] = len(self.connections)
        
        logger.info(f"[+] 新连接: {client_id} (Session: {session_id})")
        logger.info(f"    当前连接数: {self.stats['active_connections']}/{self.max_connections}")
        
        try:
            async for message in websocket:
                await self._process_message(websocket, message, session_id)
        except ConnectionClosed as e:
            logger.info(f"[-] 连接关闭: {client_id} (Code: {e.code})")
        except Exception as e:
            logger.error(f"[!] 处理错误: {client_id} - {e}")
        finally:
            # 清理会话
            if session_id in self.sessions:
                session = self.sessions[session_id]
                stats = session.get_stats()
                logger.info(f"    会话统计: {stats['audio_duration']:.2f}s音频, {stats['duration']:.2f}s时长")
                del self.sessions[session_id]
            
            self.connections.discard(websocket)
            self.stats["active_connections"] = len(self.connections)
            logger.info(f"[-] 断开连接: {client_id}")
    
    async def _process_message(self, websocket, message: str, session_id: str):
        """处理客户端消息.
        
        Args:
            websocket: WebSocket连接
            message: 消息内容
            session_id: 会话ID
        """
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type == "config":
                await self._handle_config(websocket, data, session_id)
            elif msg_type == "audio":
                await self._handle_audio(websocket, data, session_id)
            elif msg_type == "end":
                await self._handle_end(websocket, data, session_id)
            elif msg_type == "hotword":
                await self._handle_hotword(websocket, data)
            else:
                logger.warning(f"未知消息类型: {msg_type}")
                
        except json.JSONDecodeError:
            logger.error(f"无效的JSON: {message[:100]}")
        except Exception as e:
            logger.error(f"处理消息错误: {e}")
    
    async def _handle_hotword(self, websocket, data: dict):
        """处理热词管理指令.
        
        Args:
            websocket: WebSocket连接
            data: 消息数据
        """
        action = data.get("action", "")
        
        try:
            if action == "add":
                word = data.get("word", "")
                weight = data.get("weight", 1.0)
                if word:
                    self.hotword_manager.add_hotword(word, weight)
                    await websocket.send(json.dumps({
                        "type": "event",
                        "event_type": "hotword_added",
                        "data": {"word": word, "weight": weight}
                    }))
            
            elif action == "remove":
                word = data.get("word", "")
                if self.hotword_manager.remove_hotword(word):
                    await websocket.send(json.dumps({
                        "type": "event",
                        "event_type": "hotword_removed",
                        "data": {"word": word}
                    }))
            
            elif action == "list":
                hotwords = self.hotword_manager.get_hotwords()
                await websocket.send(json.dumps({
                    "type": "event",
                    "event_type": "hotword_list",
                    "data": {"hotwords": hotwords}
                }))
            
            elif action == "reload":
                self.hotword_manager.reload()
                await websocket.send(json.dumps({
                    "type": "event",
                    "event_type": "hotword_reloaded",
                    "data": self.hotword_manager.get_stats()
                }))
            
            elif action == "enable":
                enabled = data.get("enabled", True)
                self.hotword_manager.set_enabled(enabled)
                await websocket.send(json.dumps({
                    "type": "event",
                    "event_type": "hotword_status",
                    "data": self.hotword_manager.get_stats()
                }))
            
            elif action == "get_context":
                context = self.hotword_manager.get_context()
                await websocket.send(json.dumps({
                    "type": "event",
                    "event_type": "hotword_context",
                    "data": {"context": context}
                }))
            
            else:
                logger.warning(f"未知热词操作: {action}")
        
        except Exception as e:
            logger.error(f"处理热词指令错误: {e}")
    
    async def _handle_config(self, websocket, data: dict, session_id: str):
        """处理配置消息.
        
        Args:
            websocket: WebSocket连接
            data: 配置数据
            session_id: 会话ID
        """
        config_data = data.get("data", {})
        
        # 创建会话
        session = ASRSession(session_id, self.engine, config_data, self.hotword_manager)
        self.sessions[session_id] = session
        self.stats["total_sessions"] += 1
        
        logger.info(f"    [配置] 会话创建: {session_id}")
        logger.info(f"    语言: {session.language}")
        logger.info(f"    热词: {'启用' if session.hotword_manager.is_enabled() else '禁用'}")
        
        # 发送就绪响应
        response = {
            "type": "event",
            "event_type": "config_received",
            "timestamp_ms": int(time.time() * 1000),
            "data": {
                "session_id": session_id,
                "status": "ready",
                "model_info": self.engine.get_model_info(),
                "hotword_stats": self.hotword_manager.get_stats()
            }
        }
        await websocket.send(json.dumps(response))
    
    async def _handle_audio(self, websocket, data: dict, session_id: str):
        """处理音频消息.
        
        Args:
            websocket: WebSocket连接
            data: 音频数据
            session_id: 会话ID
        """
        if session_id not in self.sessions:
            logger.warning(f"无效的会话: {session_id}")
            return
        
        session = self.sessions[session_id]
        seq = data.get("seq", 0)
        audio_b64 = data.get("data", "")
        
        try:
            audio_data = base64.b64decode(audio_b64)
            session.add_audio(audio_data)
            
            # 每60个包（约6秒音频）进行一次中间识别
            if seq > 0 and seq % 60 == 0:
                # 使用线程池执行识别，不阻塞事件循环
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.executor,
                    session.recognize,
                    False
                )
                
                if result and "error" not in result:
                    # 应用ITN转换
                    text = session._apply_itn(result["text"])
                    response = {
                        "type": "result",
                        "code": 0,
                        "message": "success",
                        "data": {
                            "utterance_id": session_id,
                            "is_final": False,
                            "text": text,
                            "confidence": result.get("confidence", 0.95),
                            "start_time_ms": 0,
                            "end_time_ms": seq * 100
                        }
                    }
                    await websocket.send(json.dumps(response))
                    logger.info(f"    [识别] Seq {seq}: {text[:50]}...")
                    
        except Exception as e:
            logger.error(f"处理音频错误: {e}")
    
    async def _handle_end(self, websocket, data: dict, session_id: str):
        """处理结束消息.
        
        Args:
            websocket: WebSocket连接
            data: 结束消息
            session_id: 会话ID
        """
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        seq = data.get("seq", 0)
        
        logger.info(f"    [结束] Seq: {seq}")
        
        # 最终识别：使用线程池执行
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            session.recognize,
            True
        )
        
        if result and "error" not in result:
            # 应用ITN转换
            text = session._apply_itn(result["text"])
            response = {
                "type": "result",
                "code": 0,
                "message": "success",
                "data": {
                    "utterance_id": session_id,
                    "is_final": True,
                    "text": text,
                    "confidence": result.get("confidence", 0.95),
                    "start_time_ms": 0,
                    "end_time_ms": seq * 100
                }
            }
            await websocket.send(json.dumps(response))
            logger.info(f"    [最终结果] {text}")
            self.stats["total_recognitions"] += 1
        
        # 打印统计
        stats = session.get_stats()
        logger.info(f"    [统计] 音频: {stats['audio_duration']:.2f}s, "
                   f"会话: {stats['duration']:.2f}s")
    
    async def start(self):
        """启动服务器."""
        self._running = True
        
        # 打印启动信息
        model_info = self.engine.get_model_info()
        print("=" * 60)
        print("Qwen3-ASR 语音识别服务端")
        print("=" * 60)
        print(f"模型: {model_info['model_name']}")
        print(f"设备: {model_info['device']}")
        print(f"数据类型: {model_info['dtype']}")
        print(f"参数量: {model_info['parameters']}")
        print(f"批处理大小: {model_info['max_inference_batch_size']}")
        print("-" * 60)
        print(f"监听地址: ws://{self.host}:{self.port}")
        print(f"最大连接: {self.max_connections}")
        print("=" * 60)
        print("按 Ctrl+C 停止服务器\n")
        
        # 启动WebSocket服务器
        async with websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            ping_interval=30,
            ping_timeout=10
        ):
            try:
                await asyncio.Future()  # 永久运行
            except asyncio.CancelledError:
                pass
    
    def stop(self):
        """停止服务器."""
        self._running = False
        logger.info("服务器停止")
        
        # 关闭线程池
        self.executor.shutdown(wait=True)
        
        # 打印统计
        print("\n" + "=" * 60)
        print("运行统计:")
        print(f"  总连接数: {self.stats['total_connections']}")
        print(f"  总会话数: {self.stats['total_sessions']}")
        print(f"  总识别数: {self.stats['total_recognitions']}")
        print("=" * 60)


def load_config(config_path: str) -> dict:
    """加载配置文件.
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    default_config = {
        "model": {
            "name": "Qwen3-ASR-1.7B",
            "device": "auto",
            "dtype": "bfloat16",
            "max_inference_batch_size": 32,
            "max_new_tokens": 256,
            "forced_aligner": None,
        },
        "server": {
            "host": "0.0.0.0",
            "port": 8080,
            "max_connections": 10
        },
        "audio": {
            "sample_rate": 16000,
            "chunk_duration": 0.1
        }
    }
    
    if Path(config_path).exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            # 合并默认配置
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if sub_key not in config[key]:
                            config[key][sub_key] = sub_value
            return config
    
    return default_config


def main():
    """主函数."""
    parser = argparse.ArgumentParser(
        description="Qwen3-ASR 语音识别服务端",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置启动
  python server.py
  
  # 指定配置文件
  python server.py --config config.yaml
  
  # 指定模型和设备
  python server.py --model Qwen/Qwen3-ASR-0.6B --device cuda --port 8080
        """
    )
    
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="配置文件路径 (默认: config.yaml)"
    )
    parser.add_argument(
        "--model",
        help="模型名称 (覆盖配置文件)"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        help="运行设备 (覆盖配置文件)"
    )
    parser.add_argument(
        "--port",
        type=int,
        help="监听端口 (覆盖配置文件)"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 命令行参数覆盖配置
    if args.model:
        config["model"]["name"] = args.model
    if args.device:
        config["model"]["device"] = args.device
    if args.port:
        config["server"]["port"] = args.port
    
    # 创建服务器
    server = ASRServer(config)
    
    # 处理信号
    def signal_handler(sig, frame):
        print("\n[!] 接收到停止信号")
        server.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 启动服务器
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
