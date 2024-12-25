""" import cupy as cp

# Kiểm tra thiết bị hiện tại
try:
    with cp.cuda.Device() as device:
        device_id = device.id
        device_name = cp.cuda.runtime.getDeviceProperties(device_id)["name"]
        print("Current device ID:", device_id)
        print("Device name:", device_name)

    # Kiểm tra số lượng GPU
    num_gpus = cp.cuda.runtime.getDeviceCount()
    print("Number of GPUs available:", num_gpus)

except cp.cuda.runtime.CUDARuntimeError as e:
    print("CUDA runtime error:", e) """

import numpy as np
import cupy as cp
import time

# Tạo một mảng ngẫu nhiên kích thước lớn
N = 1000000
a = np.random.random(N)

# Thử nghiệm với CPU
start_cpu = time.time()
cpu_result = np.sqrt(a)  # Phép toán trên CPU
end_cpu = time.time()

# Thử nghiệm với GPU
start_gpu = time.time()
a_gpu = cp.asarray(a)  # Chuyển dữ liệu sang GPU
gpu_result = cp.sqrt(a_gpu)  # Phép toán trên GPU
end_gpu = time.time()

# In kết quả
print(f"Thời gian thực hiện trên CPU: {end_cpu - start_cpu:.6f} giây")
print(f"Thời gian thực hiện trên GPU: {end_gpu - start_gpu:.6f} giây")

# Nếu muốn chuyển kết quả từ GPU về CPU để kiểm tra
gpu_result_cpu = gpu_result.get()

