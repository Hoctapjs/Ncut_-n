import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
import cv2
import matplotlib.pyplot as plt

def downsample_image(image, scale=0.2):
    """
    Giảm kích thước ảnh để tăng tốc độ xử lý.
    Args:
        image: Ảnh đầu vào.
        scale: Tỷ lệ giảm kích thước (0 < scale <= 1).
    Returns:
        resized_image: Ảnh sau khi giảm kích thước.
    """
    height, width = int(image.shape[0] * scale), int(image.shape[1] * scale)
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return resized_image

def construct_graph(image, sigma_color=0.05, sigma_spatial=15):
    """
    Xây dựng đồ thị từ một ảnh.
    Args:
        image: Ảnh đầu vào (grayscale hoặc RGB).
        sigma_color: Thông số kiểm soát độ nhạy cảm về màu sắc.
        sigma_spatial: Thông số kiểm soát độ nhạy cảm không gian.
    Returns:
        W: Ma trận trọng số đồ thị (sparse matrix).
    """
    height, width, _ = image.shape
    n = height * width
    coords = np.array([[x, y] for x in range(width) for y in range(height)])
    
    # Tạo ma trận trọng số
    W = sp.lil_matrix((n, n))
    for i in range(n):
        for j in range(i+1, n):
            p1, p2 = coords[i], coords[j]
            color_diff = np.linalg.norm(image[p1[1], p1[0]] - image[p2[1], p2[0]])
            spatial_diff = np.linalg.norm(p1 - p2)
            
            if spatial_diff < sigma_spatial:  # Chỉ nối các điểm gần nhau
                weight = np.exp(-color_diff**2 / (2 * sigma_color**2)) * \
                         np.exp(-spatial_diff**2 / (2 * sigma_spatial**2))
                W[i, j] = W[j, i] = weight
                
    return W

def normalized_laplacian(W):
    """
    Tính ma trận Laplacian chuẩn hóa.
    Args:
        W: Ma trận trọng số đồ thị.
    Returns:
        L: Ma trận Laplacian chuẩn hóa.
    """
    d = np.array(W.sum(axis=1)).flatten()
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(d))
    L = sp.identity(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt
    return L

def compute_eigenvectors(L, k=2):
    """
    Tính các vectơ riêng nhỏ nhất của ma trận Laplacian chuẩn hóa.
    Args:
        L: Ma trận Laplacian chuẩn hóa.
        k: Số lượng vectơ riêng cần tính.
    Returns:
        eigenvectors: Các vectơ riêng nhỏ nhất.
    """
    eigenvalues, eigenvectors = eigsh(L, k=k, which='SM', tol=1e-5, maxiter=70000)
    return eigenvectors

def segment_image(eigenvectors, height, width, num_segments=2):
    """
    Phân đoạn ảnh dựa trên các vectơ riêng.
    Args:
        eigenvectors: Các vectơ riêng nhỏ nhất.
        height, width: Kích thước ảnh gốc.
        num_segments: Số vùng cần phân đoạn.
    Returns:
        segmented_image: Ảnh phân đoạn.
    """
    kmeans = KMeans(n_clusters=num_segments, random_state=0)
    labels = kmeans.fit_predict(eigenvectors)
    segmented_image = labels.reshape((height, width))
    return segmented_image

def normalized_cut_segmentation(image_path, num_segments=2, scale=0.5):
    """
    Hàm chính để phân đoạn ảnh sử dụng thuật toán Normalized Cut.
    Args:
        image_path: Đường dẫn đến ảnh đầu vào.
        num_segments: Số lượng vùng cần phân đoạn.
        scale: Tỷ lệ giảm kích thước ảnh.
    Returns:
        segmented_image: Ảnh sau phân đoạn.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Giảm kích thước ảnh
    image = downsample_image(image, scale)
    height, width, _ = image.shape
    
    print("Xây dựng đồ thị...")
    W = construct_graph(image)
    print("Tính ma trận Laplacian chuẩn hóa...")
    L = normalized_laplacian(W)
    print("Tính vectơ riêng...")
    eigenvectors = compute_eigenvectors(L, k=num_segments)
    print("Phân đoạn ảnh...")
    segmented_image = segment_image(eigenvectors, height, width, num_segments)
    
    return image, segmented_image

if __name__ == "__main__":
    image_path = "apple2.jpg"  # Đường dẫn đến ảnh của bạn
    num_segments = 3  # Số vùng cần phân đoạn
    scale = 0.2  # Giảm kích thước ảnh xuống 10%
    
    print("Bắt đầu phân đoạn ảnh...")
    original_image, segmented_image = normalized_cut_segmentation(image_path, num_segments, scale)
    
    # Hiển thị ảnh gốc và ảnh phân đoạn
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(original_image)
    axes[0].set_title("Ảnh Gốc")
    axes[0].axis("off")
    
    axes[1].imshow(segmented_image, cmap='tab10')
    axes[1].set_title("Ảnh Phân Đoạn")
    axes[1].axis("off")
    
    plt.show()
