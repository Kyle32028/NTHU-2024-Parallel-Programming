#!/bin/bash

# 創建輸出目錄（如果不存在）
mkdir -p ./output
mkdir -p ./cuda_output

# 創建計時結果文件
echo "Image,Sequential Time (ms),CUDA Time (ms)" > timing_comparison.csv

# 處理每個圖片
for img in ./testcases/*.png; do
    # 獲取文件名（不含路徑）
    filename=$(basename "$img")
    echo "Processing $filename..."
    
    # 執行序列版本並捕獲時間
    seq_output=$(./seq "$img" "./output/$filename" 2>&1)
    seq_time=$(echo "$seq_output" | grep "Total execution time:" | awk '{print $4}')
    
    # 如果沒有得到時間，設為 N/A
    if [ -z "$seq_time" ]; then
        seq_time="N/A"
    fi
    
    # 執行 CUDA 版本並捕獲時間
    cuda_output=$(./fft "$img" "./cuda_output/$filename" 2>&1)
    cuda_time=$(echo "$cuda_output" | grep "Total execution time:" | awk '{print $4}')
    
    # 如果沒有得到時間，設為 N/A
    if [ -z "$cuda_time" ]; then
        cuda_time="N/A"
    fi
    
    # 將結果寫入 CSV 文件
    echo "$filename,$seq_time,$cuda_time" >> timing_comparison.csv
    
    # 在控制台顯示進度
    echo "  Sequential: $seq_time ms"
    echo "  CUDA: $cuda_time ms"
    echo "----------------------------------------"
done

echo "完成！結果已保存到 timing_comparison.csv"