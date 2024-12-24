import cupy as cp

# Kiểm tra GPU khả dụng
print("Có GPU không:", cp.cuda.is_available())

# Kiểm tra phiên bản CUDA
print("Phiên bản CUDA:", cp.cuda.runtime.runtimeGetVersion())
