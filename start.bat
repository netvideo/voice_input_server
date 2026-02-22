@echo off
chcp 65001 >nul
title Qwen3-ASR 服务端
cls

echo ========================================
echo    Qwen3-ASR 语音识别服务端
echo ========================================
echo.
echo  1. 下载模型
echo  2. 启动服务端 (GPU/自动)
echo  3. 启动服务端 (CPU模式)
echo  4. 查看配置
echo  5. 退出
echo.
echo ========================================

set /p choice=请选择 (1-5): 

if "%choice%"=="1" goto download
if "%choice%"=="2" goto start_gpu
if "%choice%"=="3" goto start_cpu
if "%choice%"=="4" goto config
if "%choice%"=="5" goto end

echo 无效选择，请重新运行
goto end

:download
echo.
echo 正在下载模型...
python download_model.py --model Qwen/Qwen3-ASR-1.7B
goto end

:start_gpu
echo.
echo 正在启动服务端 (自动检测设备)...
python server.py --config config.yaml
goto end

:start_cpu
echo.
echo 正在启动服务端 (CPU模式)...
python server.py --device cpu --config config.yaml
goto end

:config
echo.
echo 当前配置:
echo ----------------------------------------
type config.yaml
echo ----------------------------------------
pause
goto end

:end
echo.
echo 按任意键退出...
pause >nul
