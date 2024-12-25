import cupy as cp
import numpy as np
import scipy.sparse as sp
from cuml.cluster import KMeans as cuKMeans
from sklearn.preprocessing import normalize
import cv2
import matplotlib.pyplot as plt

def downsample_image(image, scale):
    """Giảm kích thước ảnh để tăng tốc độ xử lý."""
    height, width = int(image.shape[0] * scale), int(image.shape[1] * scale)
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return resized_image

def construct_graph(image, sigma_color, sigma_spatial):
    """Xây dựng đồ thị từ ảnh với hiệu năng cao."""
    height, width, _ = image.shape
    n = height * width
    coords = cp.array([[x, y] for x in range(width) for y in range(height)])
    colors = cp.array(image.reshape(-1, 3))

    # Tính khoảng cách không gian và màu trên GPU
    distances_spatial = cp.linalg.norm(coords[:, None] - coords, axis=-1)
    distances_color = cp.linalg.norm(colors[:, None] - colors, axis=-1)

    # Tính trọng số
    W = cp.exp(-distances_color**2 / (2 * sigma_color**2)) * cp.exp(-distances_spatial**2 / (2 * sigma_spatial**2))
    W[distances_spatial > sigma_spatial] = 0  # Loại bỏ các cạnh xa

    return sp.csr_matrix(W.get())  # Chuyển đổi về dạng sparse cho scipy

def normalized_laplacian(W):
    """Tính ma trận Laplacian chuẩn hóa."""
    d = cp.array(W.sum(axis=1)).flatten()
    D_inv_sqrt = cp.diag(1.0 / cp.sqrt(d))
    W_dense = cp.array(W.toarray())  # Chuyển đổi ma trận W từ sparse sang dense
    L = cp.eye(W_dense.shape[0]) - D_inv_sqrt @ W_dense @ D_inv_sqrt
    return L

def compute_eigenvectors(L, k=2):
    """Tính các vectơ riêng nhỏ nhất của ma trận Laplacian chuẩn hóa."""
    L_gpu = cp.array(L)  # Giữ ma trận trên GPU
    eigenvalues, eigenvectors = cp.linalg.eigh(L_gpu)  # Tính eigenvectors trên GPU
    return eigenvectors[:, :k]  # Trả về k vectơ riêng nhỏ nhất

def segment_image(eigenvectors, height, width, num_segments):
    """Phân đoạn ảnh dựa trên các vectơ riêng."""
    eigenvectors_normalized = normalize(eigenvectors, axis=1)
    kmeans = cuKMeans(n_clusters=num_segments, random_state=0)  # Sử dụng cuML cho KMeans
    labels = kmeans.fit_predict(eigenvectors_normalized)
    segmented_image = labels.reshape((height, width))
    return segmented_image

def normalized_cut_segmentation(image_path, num_segments, scale, sigma_color, sigma_spatial):
    """Phân đoạn ảnh sử dụng thuật toán Normalized Cut."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Giảm kích thước ảnh
    image = downsample_image(image, scale)
    height, width, _ = image.shape

    print("Xây dựng đồ thị...")
    W = construct_graph(image, sigma_color, sigma_spatial)
    print("Tính ma trận Laplacian chuẩn hóa...")
    L = normalized_laplacian(W)
    print("Tính vectơ riêng...")
    eigenvectors = compute_eigenvectors(L, k=num_segments)
    print("Phân đoạn ảnh...")
    segmented_image = segment_image(eigenvectors, height, width, num_segments)

    return image, segmented_image

if __name__ == "__main__":
    image_path = "apple2.jpg"  # Đường dẫn đến ảnh của bạn
    num_segments = 4  # Số vùng cần phân đoạn
    scale = 0.3  # Tỷ lệ giảm kích thước
    sigma_color = 0.2  # Độ nhạy màu
    sigma_spatial = 20  # Độ nhạy không gian

    print("Bắt đầu phân đoạn ảnh...")
    original_image, segmented_image = normalized_cut_segmentation(
        image_path, num_segments, scale, sigma_color, sigma_spatial)

    # Hiển thị ảnh gốc và ảnh phân đoạn
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(original_image)
    axes[0].set_title("Ảnh Gốc")
    axes[0].axis("off")

    axes[1].imshow(segmented_image, cmap='tab10')
    axes[1].set_title("Ảnh Phân Đoạn")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
