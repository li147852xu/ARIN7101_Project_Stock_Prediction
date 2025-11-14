@echo off
REM 股票短期涨跌预测项目快速启动脚本 (Windows)

echo ==========================================
echo Stock Price Movement Prediction
echo ==========================================
echo.

REM 检查Python版本
echo Checking Python version...
python --version
echo.

REM 创建虚拟环境（如果不存在）
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo Virtual environment created.
    echo.
)

REM 激活虚拟环境
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM 安装依赖
echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt
echo Dependencies installed.
echo.

REM 创建必要的目录
echo Creating directories...
if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed
if not exist "results\models" mkdir results\models
if not exist "plots" mkdir plots
if not exist "logs" mkdir logs
echo Directories created.
echo.

REM 运行主程序
echo ==========================================
echo Starting main program...
echo ==========================================
echo.

python main.py %*

echo.
echo ==========================================
echo Program finished!
echo ==========================================
pause

