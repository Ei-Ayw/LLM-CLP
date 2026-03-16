#!/bin/bash
# 清理空间脚本

echo "当前磁盘使用: $(df -h /root/lanyun-fs | tail -1 | awk '{print $5}')"

# 1. 压缩大日志
echo "压缩大日志文件..."
find logs/ -name "*.log" -size +10M ! -name "*.gz" -exec gzip {} \;

# 2. 删除中间checkpoint（保留最终模型）
echo "检查可删除的checkpoint..."
find src_result/models/ -name "*checkpoint*.pth" -o -name "*epoch*.pth" | head -5

echo "完成！新的磁盘使用: $(df -h /root/lanyun-fs | tail -1 | awk '{print $5}')"
