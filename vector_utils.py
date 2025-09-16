# vector_utils.py
"""
向量与线性代数计算器函数集
包含：
- 点积、范数 (L1, L2, p-范数)
- L1 / L2 距离
- 余弦相似度、余弦距离
- 皮尔逊相关系数
- 向量归一化（L2 / L1）
- 向量投影与正交分解
- 角度（弧度 / 度）
- 批量余弦相似度（对一组向量）
- 简易数值容错（避免除零）
使用说明：
    import numpy as np
    from vector_utils import *
保存文件后可直接导入使用。
"""

from typing import Iterable, Tuple, Optional
import math
import numpy as np

# -------------------------
# 辅助函数
# -------------------------

def _to_np(x) -> np.ndarray:
    """
    将输入转换为 1-D numpy 数组。
    支持 list/tuple/numpy array/scalar（标量会转成 1 元素数组）。
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr

def _check_same_shape(a: np.ndarray, b: np.ndarray):
    """
    验证 a 和 b 的形状匹配，抛出 ValueError 否则。
    """
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")

def _safe_div(a: float, b: float, eps: float = 1e-12) -> float:
    """
    安全除法：如果分母接近 0，返回 0（或用 eps 避免 NaN/inf）。
    """
    if abs(b) < eps:
        return 0.0
    return a / b

# -------------------------
# 基本运算
# -------------------------

def dot(a: Iterable, b: Iterable) -> float:
    """
    计算两向量的点积（内积）。
    """
    aa = _to_np(a)
    bb = _to_np(b)
    _check_same_shape(aa, bb)
    return float(np.dot(aa, bb))

def norm(a: Iterable, p: float = 2.0) -> float:
    """
    计算向量的 p-范数（默认 L2 范数）。
    p 可以是 1, 2, 或任意正数。p == np.inf 支持最大绝对值范数。
    """
    aa = _to_np(a)
    if p == np.inf:
        return float(np.max(np.abs(aa)))
    if p <= 0:
        raise ValueError("p must be > 0 or np.inf")
    return float(np.sum(np.abs(aa) ** p) ** (1.0 / p))

def l2_norm(a: Iterable) -> float:
    """L2 范数快捷函数。"""
    return norm(a, p=2.0)

def l1_norm(a: Iterable) -> float:
    """L1 范数快捷函数。"""
    return norm(a, p=1.0)

# -------------------------
# 距离
# -------------------------

def l2_distance(a: Iterable, b: Iterable) -> float:
    """
    计算两向量的 L2 (Euclidean) 距离。
    """
    aa = _to_np(a)
    bb = _to_np(b)
    _check_same_shape(aa, bb)
    diff = aa - bb
    return float(np.linalg.norm(diff))

def l1_distance(a: Iterable, b: Iterable) -> float:
    """
    计算两向量的 L1 (Manhattan) 距离。
    """
    aa = _to_np(a)
    bb = _to_np(b)
    _check_same_shape(aa, bb)
    return float(np.sum(np.abs(aa - bb)))

def p_distance(a: Iterable, b: Iterable, p: float) -> float:
    """
    计算两向量的 p-范数距离（Lp distance）。
    """
    aa = _to_np(a)
    bb = _to_np(b)
    _check_same_shape(aa, bb)
    if p <= 0:
        raise ValueError("p must be > 0")
    return float(np.sum(np.abs(aa - bb) ** p) ** (1.0 / p))

# -------------------------
# 余弦相似度 / 余弦距离
# -------------------------

def cosine_similarity(a: Iterable, b: Iterable, eps: float = 1e-12) -> float:
    """
    计算余弦相似度：
        cos_sim = (a · b) / (||a|| * ||b||)
    返回值范围 [-1, 1]。若其中一个向量全零，则返回 0（通过 eps 安全处理）。
    """
    aa = _to_np(a)
    bb = _to_np(b)
    _check_same_shape(aa, bb)
    denom = np.linalg.norm(aa) * np.linalg.norm(bb)
    if denom < eps:
        return 0.0
    return float(np.dot(aa, bb) / denom)

def cosine_distance(a: Iterable, b: Iterable, eps: float = 1e-12) -> float:
    """
    余弦距离（常见定义）：
        cosine_distance = 1 - cosine_similarity
    范围 [0, 2]（当使用无归一化向量，两个方向相反时为 2；标准范围为 0..2）
    """
    return 1.0 - cosine_similarity(a, b, eps=eps)

# -------------------------
# 角度与投影
# -------------------------

def angle_between(a: Iterable, b: Iterable, in_degrees: bool = False, eps: float = 1e-12) -> float:
    """
    返回向量 a 与 b 之间的夹角（弧度或角度）。
    用 acos(clamp(cosine, -1, 1)) 避免数值问题。
    若任一向量为 0，则返回 0。
    """
    cos = cosine_similarity(a, b, eps=eps)
    cos = max(-1.0, min(1.0, cos))
    ang = math.acos(cos)
    if in_degrees:
        return math.degrees(ang)
    return ang

def project(a: Iterable, b: Iterable) -> np.ndarray:
    """
    将 a 投影到 b 上（向量投影），返回投影向量：
        proj_b(a) = ( (a·b) / (b·b) ) * b
    若 b 为零向量，返回与 b 同形状的零向量。
    """
    aa = _to_np(a)
    bb = _to_np(b)
    _check_same_shape(aa, bb)
    bb_norm_sq = np.dot(bb, bb)
    if bb_norm_sq == 0:
        return np.zeros_like(bb)
    coef = np.dot(aa, bb) / bb_norm_sq
    return coef * bb

def orthogonal_component(a: Iterable, b: Iterable) -> np.ndarray:
    """
    返回向量 a 在 b 的正交分量（a - proj_b(a)）。
    """
    aa = _to_np(a)
    bb = _to_np(b)
    return aa - project(aa, bb)

# -------------------------
# 相关系数
# -------------------------

def pearson_correlation(a: Iterable, b: Iterable, eps: float = 1e-12) -> float:
    """
    计算皮尔逊相关系数：
        cov(a,b) / (std(a) * std(b))
    若标准差接近 0，则返回 0。
    """
    aa = _to_np(a).astype(float)
    bb = _to_np(b).astype(float)
    _check_same_shape(aa, bb)
    if aa.size == 0:
        raise ValueError("empty arrays")
    a_mean = aa.mean()
    b_mean = bb.mean()
    a_center = aa - a_mean
    b_center = bb - b_mean
    denom = np.sqrt(np.sum(a_center ** 2) * np.sum(b_center ** 2))
    if denom < eps:
        return 0.0
    return float(np.sum(a_center * b_center) / denom)

# -------------------------
# 归一化
# -------------------------

def normalize(a: Iterable, p: float = 2.0, eps: float = 1e-12) -> np.ndarray:
    """
    将向量按 p-范数归一化（默认 L2 归一化），返回 numpy 向量。
    若范数接近 0，则返回零向量（数值安全）。
    """
    aa = _to_np(a)
    n = norm(aa, p=p)
    if n < eps:
        return np.zeros_like(aa)
    return aa / n

def l2_normalize(a: Iterable, eps: float = 1e-12) -> np.ndarray:
    """L2 归一化快捷函数。"""
    return normalize(a, p=2.0, eps=eps)

def l1_normalize(a: Iterable, eps: float = 1e-12) -> np.ndarray:
    """L1 归一化快捷函数。"""
    return normalize(a, p=1.0, eps=eps)

# -------------------------
# 批量/矩阵操作（用于大量向量比较）
# -------------------------

def pairwise_cosine_similarity_matrix(X: np.ndarray, Y: Optional[np.ndarray] = None, eps: float = 1e-12) -> np.ndarray:
    """
    计算矩阵之间的逐对余弦相似度。
    输入：
        X: shape (n_samples_x, n_features)
        Y: shape (n_samples_y, n_features) 或 None（若 None，则 Y = X）
    返回：
        matrix shape (n_samples_x, n_samples_y)
    注意：会对每行进行 L2 归一化以加速计算。
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D array (n_samples, n_features)")
    if Y is None:
        Y = X
    else:
        Y = np.asarray(Y, dtype=float)
        if Y.ndim != 2:
            raise ValueError("Y must be 2D array")

    if X.shape[1] != Y.shape[1]:
        raise ValueError("feature dimension mismatch between X and Y")

    # 归一化每行（L2），避免除0
    X_norms = np.linalg.norm(X, axis=1, keepdims=True)
    Y_norms = np.linalg.norm(Y, axis=1, keepdims=True)
    X_safe = X.copy()
    Y_safe = Y.copy()
    X_safe[X_norms[:, 0] < eps] = 0.0
    Y_safe[Y_norms[:, 0] < eps] = 0.0
    Xn = X_safe / np.maximum(X_norms, eps)
    Yn = Y_safe / np.maximum(Y_norms, eps)
    return np.dot(Xn, Yn.T)

def pairwise_l2_distance_matrix(X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
    """
    计算矩阵之间的逐对 L2 距离。
    使用广播或高效公式 (a-b)^2 = a^2 + b^2 - 2ab。
    返回 shape (n_x, n_y)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if Y is None:
        Y = X
    else:
        Y = np.asarray(Y, dtype=float)
        if Y.ndim != 2:
            raise ValueError("Y must be 2D")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("feature dimension mismatch")

    # 计算平方和并使用矩阵运算
    xx = np.sum(X ** 2, axis=1)[:, np.newaxis]  # shape (n_x, 1)
    yy = np.sum(Y ** 2, axis=1)[np.newaxis, :]  # shape (1, n_y)
    xy = np.dot(X, Y.T)                          # shape (n_x, n_y)
    # 距离矩阵（数值误差可能导致小负数，取最大与0）
    dist_sq = np.maximum(xx + yy - 2.0 * xy, 0.0)
    return np.sqrt(dist_sq)

# -------------------------
# 其他工具
# -------------------------

def is_zero_vector(a: Iterable, eps: float = 1e-12) -> bool:
    """判断向量是否为零向量（按 L2 范数）。"""
    return l2_norm(a) < eps

def clip_cosine(x: float) -> float:
    """将浮点数限制到 [-1,1]，常用于 acos 前的保护。"""
    return max(-1.0, min(1.0, float(x)))

# -------------------------
# 小示例（作为模块被直接运行时展示）
# -------------------------
if __name__ == "__main__":
    # 简单示例，供快速验证
    a = [1, 2, 3]
    b = [4, 5, 6]

    print("a · b =", dot(a, b))
    print("||a|| =", l2_norm(a))
    print("L1 distance =", l1_distance(a, b))
    print("L2 distance =", l2_distance(a, b))
    print("cosine similarity =", cosine_similarity(a, b))
    print("angle (deg) =", angle_between(a, b, in_degrees=True))

    X = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
    print("pairwise cosine:\n", pairwise_cosine_similarity_matrix(X))
    print("pairwise L2:\n", pairwise_l2_distance_matrix(X))
