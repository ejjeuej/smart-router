#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import random
import matplotlib.pyplot as plt
import numpy as np

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 冒泡排序
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

# 选择排序
def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

# 插入排序
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

# 快速排序
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

# 归并排序
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# 堆排序
def heap_sort(arr):
    def heapify(arr, n, i):
        largest = i
        l = 2 * i + 1
        r = 2 * i + 2
        if l < n and arr[i] < arr[l]:
            largest = l
        if r < n and arr[largest] < arr[r]:
            largest = r
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            heapify(arr, n, largest)
    
    n = len(arr)
    for i in range(n, -1, -1):
        heapify(arr, n, i)
    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)
    return arr

# 性能测试函数
def measure_time(sort_func, arr):
    start_time = time.time()
    sort_func(arr.copy())
    end_time = time.time()
    return end_time - start_time

# 生成测试数据
def generate_data(size, data_type='random'):
    if data_type == 'random':
        return [random.randint(0, 1000) for _ in range(size)]
    elif data_type == 'sorted':
        return list(range(size))
    elif data_type == 'reverse':
        return list(range(size, 0, -1))
    elif data_type == 'partial':
        arr = list(range(size))
        for i in range(size // 10):
            j = random.randint(0, size-1)
            k = random.randint(0, size-1)
            arr[j], arr[k] = arr[k], arr[j]
        return arr

# 测试不同算法在不同数据规模下的性能
algorithms = {
    'Bubble Sort': bubble_sort,
    'Selection Sort': selection_sort,
    'Insertion Sort': insertion_sort,
    'Quick Sort': quick_sort,
    'Merge Sort': merge_sort,
    'Heap Sort': heap_sort
}

# 定义测试数据规模
sizes = [100, 500, 1000, 2000, 3000]
data_types = ['random', 'sorted', 'reverse', 'partial']

# 存储结果
results = {alg: {dt: [] for dt in data_types} for alg in algorithms}

# 执行测试
print("开始测试排序算法性能...")
for size in sizes:
    print(f"测试数据规模: {size}")
    for dt in data_types:
        data = generate_data(size, dt)
        for name, func in algorithms.items():
            # 对于大数据集，跳过时间复杂度较高的算法
            if size > 1000 and name in ['Bubble Sort', 'Selection Sort']:
                results[name][dt].append(None)
                continue
            
            elapsed_time = measure_time(func, data)
            results[name][dt].append(elapsed_time)
            print(f"  {name} 在 {dt} 数据上的执行时间: {elapsed_time:.6f}s")

# 绘制性能对比图
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for idx, dt in enumerate(data_types):
    ax = axes[idx]
    for name, times in results.items():
        # 过滤掉None值
        valid_times = [t for t in times[dt] if t is not None]
        valid_sizes = sizes[:len(valid_times)]
        ax.plot(valid_sizes, valid_times, marker='o', label=name)
    
    ax.set_xlabel('数据规模')
    ax.set_ylabel('执行时间 (秒)')
    ax.set_title(f'{dt.capitalize()} 数据性能对比')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig('sorting_performance_comparison.png')
plt.show()

print("\n排序算法时间复杂度总结:")
print("| 算法 | 最好情况 | 平均情况 | 最坏情况 | 空间复杂度 |")
print("|------|----------|----------|----------|------------|")
print("| 冒泡排序 | O(n) | O(n²) | O(n²) | O(1) |")
print("| 选择排序 | O(n²) | O(n²) | O(n²) | O(1) |")
print("| 插入排序 | O(n) | O(n²) | O(n²) | O(1) |")
print("| 快速排序 | O(n log n) | O(n log n) | O(n²) | O(log n) |")
print("| 归并排序 | O(n log n) | O(n log n) | O(n log n) | O(n) |")
print("| 堆排序 | O(n log n) | O(n log n) | O(n log n) | O(1) |")