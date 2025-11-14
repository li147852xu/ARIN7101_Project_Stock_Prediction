#!/bin/bash
# GPU实时监控脚本

echo "GPU Monitoring Script"
echo "Press Ctrl+C to stop"
echo ""

# 检查nvidia-smi是否可用
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. Please check NVIDIA driver installation."
    exit 1
fi

# 创建日志目录
mkdir -p logs

# 日志文件
LOG_FILE="logs/gpu_monitor_$(date +%Y%m%d_%H%M%S).log"

echo "GPU monitoring started at $(date)" | tee -a $LOG_FILE
echo "Log file: $LOG_FILE"
echo ""

# 监控循环
while true; do
    clear
    
    # 显示时间
    echo "=================================="
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=================================="
    echo ""
    
    # GPU信息
    echo "=== GPU Status ==="
    nvidia-smi --query-gpu=index,name,temperature.gpu,power.draw,power.limit,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits | \
    awk -F',' '{
        printf "GPU %s: %s\n", $1, $2
        printf "  Temperature: %s°C\n", $3
        printf "  Power: %s W / %s W\n", $4, $5
        printf "  GPU Util: %s%%\n", $6
        printf "  Mem Util: %s%%\n", $7
        printf "  Memory: %s MB / %s MB\n", $8, $9
        printf "\n"
    }'
    
    # 进程信息
    echo "=== Training Processes ==="
    PROCESSES=$(ps aux | grep -E "python.*main.py|python.*train.py" | grep -v grep)
    if [ -z "$PROCESSES" ]; then
        echo "No training processes found."
    else
        echo "$PROCESSES" | awk '{printf "PID: %s  CPU: %s%%  MEM: %s%%  CMD: %s\n", $2, $3, $4, substr($0, index($0,$11))}'
    fi
    
    echo ""
    echo "=== GPU Memory Details ==="
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | \
    awk -F',' '{printf "PID: %s  Process: %s  GPU Mem: %s\n", $1, $2, $3}'
    
    # 记录到日志
    echo "$(date '+%Y-%m-%d %H:%M:%S')" >> $LOG_FILE
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,temperature.gpu --format=csv,noheader >> $LOG_FILE
    
    # 刷新间隔
    sleep 2
done

