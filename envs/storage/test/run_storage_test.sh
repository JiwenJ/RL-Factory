#!/bin/bash

# 设置持久化目录（以日期为结尾）
DATE_SUFFIX=$(date +"%Y%m%d_%H%M%S")
# PERSIST_DIR="/data1/jjw/rl_cache/${DATE_SUFFIX}"
PERSIST_DIR="/data1/jjw/rl_cache/20250810_220722"

# # 创建持久化目录
# mkdir -p "$PERSIST_DIR"

# echo "=== 第一次运行：测试持久化保存功能 ==="
# python /data1/jjw/RL-Factory/envs/storage/test/storage_test.py \
#     --cache single \
#     --persist \
#     --eviction lru \
#     --max-size 1000 \
#     --persist-dir "$PERSIST_DIR"

# echo ""
# echo "=== 检查生成的缓存文件 ==="
# ls -la "$PERSIST_DIR"/*.cache 2>/dev/null || echo "没有找到缓存文件"

# echo ""
echo "=== 第二次运行：测试缓存加载功能 ==="
# 检查是否有缓存文件
CACHE_COUNT=$(ls "$PERSIST_DIR"/*.cache 2>/dev/null | wc -l)
if [ "$CACHE_COUNT" -gt 0 ]; then
    echo "找到 $CACHE_COUNT 个缓存文件，加载整个目录: $PERSIST_DIR"
    ls -la "$PERSIST_DIR"/*.cache
    echo ""
    python /data1/jjw/RL-Factory/envs/storage/test/storage_test.py \
        --cache single \
        --persist \
        --eviction lru \
        --max-size 1000 \
        --persist-dir "$PERSIST_DIR" \
        --load-cache "$PERSIST_DIR"
else
    echo "没有找到缓存文件，跳过加载测试"
fi