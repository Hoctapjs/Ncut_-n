import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from skimage import io, color

# 1. Tính ma trận trọng số
def compute_weight_matrix(image, sigma_i=0.1, sigma_x=10):
    h, w, c = image.shape
    coords = np.array(np.meshgrid(range(h), range(w))).reshape(2, -1).T  # Tọa độ (x, y)
    features = image.reshape(-1, c)  # Đặc trưng màu
    
    # Tính độ tương đồng về đặc trưng và không gian
    W = rbf_kernel(features, gamma=1/(2 * sigma_i**2)) * rbf_kernel(coords, gamma=1/(2 * sigma_x**2))
    return W

# 2. Tính ma trận Laplace
def compute_laplacian(W):
    D = np.diag(W.sum(axis=1))  # Ma trận đường chéo
    L = D - W
    return L, D

# 3. Giải bài toán trị riêng
def compute_eigen(L, D, k=2):
    # Giải bài toán trị riêng tổng quát
    vals, vecs = eigsh(L, k=k, M=D, which='SM')  # 'SM' tìm trị riêng nhỏ nhất
    return vecs  # Trả về k vector riêng

# 4. Gán nhãn cho từng điểm ảnh dựa trên vector riêng
def assign_labels(eigen_vectors, k):
    # Dùng K-Means để gán nhãn
    kmeans = KMeans(n_clusters=k, random_state=0).fit(eigen_vectors)
    labels = kmeans.labels_
    return labels

# 5. Hiển thị kết quả
def display_segmentation(image, labels, k):
    h, w, c = image.shape
    segmented_image = np.zeros_like(image, dtype=np.uint8)
    
    # Tạo bảng màu ngẫu nhiên
    colors = np.random.randint(0, 255, size=(k, 3), dtype=np.uint8)
    
    # Tô màu từng vùng
    for i in range(k):
        segmented_image[labels.reshape(h, w) == i] = colors[i]
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Segmented Image")
    plt.imshow(segmented_image)
    plt.axis('off')
    plt.show()

# 6. Kết hợp toàn bộ
def normalized_cuts(image_path, k=2):
    # Đọc ảnh và chuẩn hóa
    image = io.imread(image_path)
    if image.ndim == 2:  # Nếu là ảnh xám, chuyển thành RGB
        image = color.gray2rgb(image)
    elif image.shape[2] == 4:  # Nếu là ảnh RGBA, loại bỏ kênh alpha
        image = image[:, :, :3]
    image = image / 255.0  # Chuẩn hóa về [0, 1]
    
    # Tính toán Ncuts
    print("Computing weight matrix...")
    W = compute_weight_matrix(image)
    
    print("Computing Laplacian...")
    L, D = compute_laplacian(W)
    
    print("Computing eigenvectors...")
    eigen_vectors = compute_eigen(L, D, k=k)  # Tính k vector riêng
    
    print("Partitioning graph...")
    labels = assign_labels(eigen_vectors, k)  # Gán nhãn cho mỗi điểm ảnh
    
    print("Displaying results...")
    display_segmentation(image, labels, k)

# 7. Chạy thử nghiệm
if __name__ == "__main__":
    # Đường dẫn tới ảnh của bạn
    image_path = "D:/result_traitao.jpg"  # Thay bằng đường dẫn ảnh của bạn
    normalized_cuts(image_path, k=3)  # Phân vùng thành 4 nhóm
