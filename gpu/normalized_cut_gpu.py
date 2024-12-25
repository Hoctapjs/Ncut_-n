import cupy as cp
from scipy.spatial import cKDTree  # CuPy không hỗ trợ KDTree, sử dụng từ SciPy
from sklearn.cluster import KMeans
import cv2
import matplotlib.pyplot as plt

def downsample_image(image, scale=1):
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

def construct_graph_gpu(image, sigma_color=0.1, sigma_spatial=10, r=5):
    """
    Xây dựng đồ thị trên GPU.
    Args:
        image: Ảnh đầu vào (grayscale hoặc RGB).
        sigma_color: Độ nhạy cảm về màu sắc.
        sigma_spatial: Độ nhạy cảm không gian.
        r: Bán kính lân cận để kết nối các điểm.
    Returns:
        W: Ma trận trọng số đồ thị (dạng sparse).
    """
    height, width, _ = image.shape
    n = height * width
    coords = cp.array([[x, y] for x in range(width) for y in range(height)], dtype=cp.float32)
    pixels = cp.array(image.reshape(-1, image.shape[2]), dtype=cp.float32)

    # Tạo cây KD trên CPU để tìm hàng xóm
    tree = cKDTree(cp.asnumpy(coords))
    neighbors = tree.query_ball_tree(tree, r=r)

    # Khởi tạo ma trận trọng số
    W = cp.zeros((n, n), dtype=cp.float32)

    for i, neighbors_i in enumerate(neighbors):
        neighbors_i = cp.array(neighbors_i, dtype=cp.int32)
        if len(neighbors_i) > 0:
            color_diff = cp.linalg.norm(pixels[i] - pixels[neighbors_i], axis=1)
            spatial_diff = cp.linalg.norm(coords[i] - coords[neighbors_i], axis=1)
            weights = cp.exp(-color_diff**2 / (2 * sigma_color**2)) * \
                      cp.exp(-spatial_diff**2 / (2 * sigma_spatial**2))
            W[i, neighbors_i] = weights

    return W

def normalized_laplacian_gpu(W):
    """
    Tính ma trận Laplacian chuẩn hóa trên GPU.
    Args:
        W: Ma trận trọng số đồ thị.
    Returns:
        L: Ma trận Laplacian chuẩn hóa.
    """
    d = W.sum(axis=1)
    D_inv_sqrt = cp.diag(1.0 / cp.sqrt(d))
    L = cp.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt
    return L

def compute_eigenvectors_gpu(L, k=2):
    """
    Tính các vectơ riêng nhỏ nhất của ma trận Laplacian trên GPU.
    Args:
        L: Ma trận Laplacian chuẩn hóa.
        k: Số lượng vectơ riêng cần tính.
    Returns:
        eigenvectors: Các vectơ riêng nhỏ nhất.
    """
    eigenvalues, eigenvectors = cp.linalg.eigh(L)  # Tính tất cả giá trị riêng
    idx = cp.argsort(eigenvalues)[:k]  # Chọn k giá trị riêng nhỏ nhất
    return eigenvectors[:, idx]

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
    eigenvectors = cp.asnumpy(eigenvectors)  # Chuyển về CPU cho KMeans
    kmeans = KMeans(n_clusters=num_segments, random_state=0)
    labels = kmeans.fit_predict(eigenvectors)
    segmented_image = labels.reshape((height, width))
    return segmented_image

def normalized_cut_segmentation_gpu(image_path, num_segments=2, scale=0.5):
    """
    Hàm chính để phân đoạn ảnh sử dụng GPU.
    Args:
        image_path: Đường dẫn đến ảnh đầu vào.
        num_segments: Số lượng vùng cần phân đoạn.
        scale: Tỷ lệ giảm kích thước ảnh.
    Returns:
        segmented_image: Ảnh sau phân đoạn.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = downsample_image(image, scale=scale)  # Giảm kích thước ảnh
    height, width, _ = image.shape

    print("Xây dựng đồ thị trên GPU...")
    W = construct_graph_gpu(image)
    print("Tính ma trận Laplacian chuẩn hóa trên GPU...")
    L = normalized_laplacian_gpu(W)
    print("Tính vectơ riêng trên GPU...")
    eigenvectors = compute_eigenvectors_gpu(L, k=num_segments)
    print("Phân đoạn ảnh...")
    segmented_image = segment_image(eigenvectors, height, width, num_segments)
    
    return segmented_image

if __name__ == "__main__":
    image_path = "example.jpg"  # Đường dẫn đến ảnh của bạn
    num_segments = 3  # Số vùng cần phân đoạn
    
    print("Bắt đầu phân đoạn ảnh với GPU...")
    segmented_image = normalized_cut_segmentation_gpu(image_path, num_segments)
    
    plt.imshow(segmented_image, cmap='tab10')
    plt.title("Ảnh sau khi phân đoạn")
    plt.axis("off")
    plt.show()
