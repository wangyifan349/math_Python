# matrix_tools.py
"""
更完整的线性代数工具集合（纯 Python + NumPy 实现）
包含（但不限于）：
- 向量操作：点积、范数、距离、余弦相似度
- 矩阵操作：转置、trace、对称、对角提取、拼接
- 分解与求解：Gaussian 消元（带/不带列主元）、LU 分解、Cholesky、QR（Gram-Schmidt 和 Householder）、SVD（调用 NumPy）、共轭梯度法
- 基本代数：行列式（递归/LU）、矩阵逆、伪逆（SVD）、秩
- 特征值/特征向量：幂法、反幂法（可选移位）
- 批量/矩阵相似度与距离计算
- 若干辅助工具与数值稳定处理
依赖：numpy
"""
from typing import Iterable, Optional, Tuple
import math
import numpy as np
# -------------------------
# 辅助与类型转换
# -------------------------
def _to_np_vec(x) -> np.ndarray:
    """转为 1-D numpy 数组"""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        arr = arr.ravel()
    return arr
def _to_np_mat(A) -> np.ndarray:
    """转为 2-D numpy 数组"""
    M = np.asarray(A, dtype=float)
    if M.ndim == 1:
        M = M.reshape(1, -1)
    if M.ndim != 2:
        raise ValueError("Input must be 2D (matrix-like)")
    return M
def _is_square(A: np.ndarray) -> bool:
    return A.ndim == 2 and A.shape[0] == A.shape[1]
# -------------------------
# 向量基础（重复/补充）
# -------------------------
def dot(a: Iterable, b: Iterable) -> float:
    a = _to_np_vec(a); b = _to_np_vec(b)
    if a.shape != b.shape:
        raise ValueError("shape mismatch")
    return float(np.dot(a, b))
def norm(a: Iterable, p: float = 2.0) -> float:
    a = _to_np_vec(a)
    if p == np.inf:
        return float(np.max(np.abs(a)))
    if p <= 0:
        raise ValueError("p must be > 0")
    return float(np.sum(np.abs(a) ** p) ** (1.0 / p))
def l2_distance(a: Iterable, b: Iterable) -> float:
    a = _to_np_vec(a); b = _to_np_vec(b)
    if a.shape != b.shape:
        raise ValueError("shape mismatch")
    return float(np.linalg.norm(a - b))
def cosine_similarity(a: Iterable, b: Iterable) -> float:
    a = _to_np_vec(a); b = _to_np_vec(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

# -------------------------
# 矩阵基础
# -------------------------

def mat_transpose(A) -> np.ndarray:
    A = _to_np_mat(A)
    return A.T

def mat_trace(A) -> float:
    A = _to_np_mat(A)
    if not _is_square(A):
        raise ValueError("trace requires square matrix")
    return float(np.trace(A))

def mat_diag(A) -> np.ndarray:
    A = _to_np_mat(A)
    return np.diag(A)

def mat_is_symmetric(A, tol=1e-12) -> bool:
    A = _to_np_mat(A)
    return np.allclose(A, A.T, atol=tol)

def mat_concat(A, B, axis=0):
    A = _to_np_mat(A); B = _to_np_mat(B)
    return np.concatenate([A, B], axis=axis)

# -------------------------
# 解线性方程：Gaussian 消元（带主元）
# -------------------------

def gaussian_elimination(A: Iterable, b: Optional[Iterable] = None, pivot: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    使用高斯消元把矩阵 A 化为上三角（行阶梯）形式；若提供 b，则同步变换 b。
    返回 (U, c) 其中 U 是上三角矩阵（浮点），c 是变换后的 b（或 None）。
    pivot=True 时使用部分列主元（行交换）。
    注意：此函数返回的是变换后的矩阵而不是解；可用于观察行简化过程或与回代配合使用。
    """
    A = _to_np_mat(A).astype(float).copy()
    n, m = A.shape
    if b is not None:
        b = _to_np_vec(b).astype(float).copy()
        if b.size != n:
            raise ValueError("b length mismatch")
    else:
        b = None

    row = 0
    for col in range(min(n, m)):
        # 选择主元行
        sel = row
        if pivot:
            # 部分主元：在 col 列中选择绝对值最大的行
            max_row = np.argmax(np.abs(A[row:, col])) + row
            sel = max_row
        # 若主元为零，跳过该列
        if abs(A[sel, col]) < 1e-15:
            continue
        # 交换行
        if sel != row:
            A[[row, sel], :] = A[[sel, row], :]
            if b is not None:
                b[[row, sel]] = b[[sel, row]]
        # 消元
        pivot_val = A[row, col]
        A[row, :] = A[row, :] / pivot_val  # 也可不归一化主行
        if b is not None:
            b[row] = b[row] / pivot_val
        for r in range(n):
            if r != row:
                factor = A[r, col]
                if factor != 0:
                    A[r, :] = A[r, :] - factor * A[row, :]
                    if b is not None:
                        b[r] = b[r] - factor * b[row]
        row += 1
        if row == n:
            break
    return A, b

def back_substitution(U: Iterable, y: Iterable) -> np.ndarray:
    """
    对上三角矩阵 U x = y 做回代求解（假设 U 为方阵且对角不为 0）。
    """
    U = _to_np_mat(U).astype(float)
    y = _to_np_vec(y).astype(float)
    n, m = U.shape
    if n != m:
        raise ValueError("U must be square")
    if y.size != n:
        raise ValueError("y length mismatch")
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        s = np.dot(U[i, i + 1:], x[i + 1:])
        if abs(U[i, i]) < 1e-15:
            raise np.linalg.LinAlgError("singular matrix in back substitution")
        x[i] = (y[i] - s) / U[i, i]
    return x

def solve_gauss(A: Iterable, b: Iterable, pivot: bool = True) -> np.ndarray:
    """
    使用高斯消元（带或不带主元）解线性方程组 Ax = b。
    """
    U, c = gaussian_elimination(A, b, pivot=pivot)
    return back_substitution(U, c)

# -------------------------
# LU 分解（Doolittle，无行列交换）
# -------------------------

def lu_decomposition(A: Iterable, pivot: bool = False) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    LU 分解：A = P L U（若 pivot=True 返回置换矩阵 P）
    若 pivot=False，返回 L, U, None（无 P）
    若 pivot=True，使用简单的部分主元并返回 P (permutation matrix)
    注意：这是基本实现，速度/稳定性低于 SciPy。
    """
    A = _to_np_mat(A).astype(float).copy()
    n, m = A.shape
    if n != m:
        raise ValueError("LU requires square matrix")
    n = A.shape[0]
    L = np.eye(n, dtype=float)
    U = A.copy()
    P = np.eye(n, dtype=float) if pivot else None

    for k in range(n):
        if pivot:
            max_row = np.argmax(np.abs(U[k:, k])) + k
            if max_row != k:
                U[[k, max_row], :] = U[[max_row, k], :]
                P[[k, max_row], :] = P[[max_row, k], :]
                if k > 0:
                    L[[k, max_row], :k] = L[[max_row, k], :k]
        if abs(U[k, k]) < 1e-15:
            raise np.linalg.LinAlgError("zero pivot encountered")
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] = U[i, k:] - L[i, k] * U[k, k:]
            U[i, k] = 0.0
    return L, U, P

def lu_solve(L: np.ndarray, U: np.ndarray, P: Optional[np.ndarray], b: Iterable) -> np.ndarray:
    b = _to_np_vec(b)
    if P is not None:
        b = P.dot(b)
    # 前代求解 L y = b
    n = L.shape[0]
    y = np.zeros(n, dtype=float)
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / (L[i, i] if L[i, i] != 0 else 1.0)
    # 回代 U x = y
    return back_substitution(U, y)

# -------------------------
# Cholesky 分解（对称正定）
# -------------------------

def cholesky_decomposition(A: Iterable) -> np.ndarray:
    """
    返回下三角 L 使得 A = L L^T（A 必须是对称正定）。
    简单实现（数值稳定性有限）。
    """
    A = _to_np_mat(A).astype(float)
    if not _is_square(A):
        raise ValueError("Cholesky requires square matrix")
    n = A.shape[0]
    L = np.zeros_like(A)
    for i in range(n):
        for j in range(i + 1):
            s = np.dot(L[i, :j], L[j, :j])
            if i == j:
                val = A[i, i] - s
                if val <= 0:
                    raise np.linalg.LinAlgError("Matrix not positive definite")
                L[i, j] = math.sqrt(val)
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]
    return L

# -------------------------
# QR 分解：Gram-Schmidt（古典）与 Householder
# -------------------------

def qr_gram_schmidt(A: Iterable) -> Tuple[np.ndarray, np.ndarray]:
    """
    经典 Gram-Schmidt 实现 QR 分解（数值不稳定）。
    返回 Q (orthonormal), R (upper triangular) 使 A = Q R
    """
    A = _to_np_mat(A).astype(float)
    n, m = A.shape
    Q = np.zeros((n, m), dtype=float)
    R = np.zeros((m, m), dtype=float)
    for j in range(m):
        v = A[:, j].copy()
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        if R[j, j] == 0:
            Q[:, j] = 0
        else:
            Q[:, j] = v / R[j, j]
    return Q, R

def qr_householder(A: Iterable) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 Householder 反射实现更稳定的 QR 分解。
    返回 Q, R 使 A = Q R
    """
    A = _to_np_mat(A).astype(float).copy()
    n, m = A.shape
    R = A.copy()
    Q = np.eye(n)
    for k in range(min(n, m)):
        x = R[k:, k]
        e1 = np.zeros_like(x); e1[0] = np.linalg.norm(x)
        v = x - e1
        if np.linalg.norm(v) == 0:
            continue
        v = v / np.linalg.norm(v)
        Hk = np.eye(n)
        Hk_k = np.eye(len(x)) - 2.0 * np.outer(v, v)
        Hk[k:, k:] = Hk_k
        R = Hk.dot(R)
        Q = Q.dot(Hk)
    return Q, R

# -------------------------
# SVD（调用 NumPy 以保证稳定与性能）与伪逆
# -------------------------

def svd_decompose(A: Iterable, full_matrices: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    调用 NumPy 返回 U, s, Vt（s 为奇异值向量）。
    """
    A = _to_np_mat(A)
    U, s, Vt = np.linalg.svd(A, full_matrices=full_matrices)
    return U, s, Vt

def pseudo_inverse(A: Iterable, tol: float = 1e-15) -> np.ndarray:
    """
    通过 SVD 计算 Moore-Penrose 伪逆。
    """
    A = _to_np_mat(A)
    U, s, Vt = svd_decompose(A, full_matrices=False)
    # 伪逆奇异值
    s_inv = np.array([1.0 / si if si > tol else 0.0 for si in s])
    return (Vt.T * s_inv).dot(U.T)

# -------------------------
# 行列式与矩阵秩（使用 LU / SVD）
# -------------------------

def determinant(A: Iterable) -> float:
    A = _to_np_mat(A)
    if not _is_square(A):
        raise ValueError("det requires square matrix")
    # 使用 LU（NumPy 的 linalg.det 更快，但这里示范）
    return float(np.linalg.det(A))

def matrix_rank(A: Iterable, tol: float = 1e-12) -> int:
    A = _to_np_mat(A)
    s = np.linalg.svd(A, compute_uv=False)
    return int(np.sum(s > tol))

# -------------------------
# 矩阵逆
# -------------------------

def matrix_inverse(A: Iterable) -> np.ndarray:
    A = _to_np_mat(A)
    if not _is_square(A):
        raise ValueError("inverse requires square matrix")
    return np.linalg.inv(A)

# -------------------------
# 幂法与反幂法（特征向量）
# -------------------------

def power_method(A: Iterable, num_iters: int = 1000, tol: float = 1e-12) -> Tuple[float, np.ndarray]:
    """
    幂法求矩阵 A 的主特征值与对应特征向量（最大模）。
    返回 (eigenvalue, eigenvector)
    """
    A = _to_np_mat(A)
    n, m = A.shape
    if n != m:
        raise ValueError("power method requires square matrix")
    b = np.random.randn(n)
    b = b / np.linalg.norm(b)
    lambda_old = 0.0
    for _ in range(num_iters):
        b_next = A.dot(b)
        if np.linalg.norm(b_next) == 0:
            return 0.0, b_next
        b_next = b_next / np.linalg.norm(b_next)
        lambda_new = float(b_next.dot(A.dot(b_next)))
        if abs(lambda_new - lambda_old) < tol:
            return lambda_new, b_next
        b = b_next
        lambda_old = lambda_new
    return lambda_new, b_next

def inverse_power_method(A: Iterable, shift: float = 0.0, num_iters: int = 1000, tol: float = 1e-12) -> Tuple[float, np.ndarray]:
    """
    反幂法（带移位）用于求接近 shift 的特征值/特征向量。
    """
    A = _to_np_mat(A)
    n, m = A.shape
    if n != m:
        raise ValueError("inverse power method requires square matrix")
    B = A - shift * np.eye(n)
    # 预先 LU 分解可加速；这里直接求逆（仅示范）
    try:
        B_inv = np.linalg.inv(B)
    except np.linalg.LinAlgError:
        raise np.linalg.LinAlgError("shifted matrix singular")
    val, vec = power_method(B_inv, num_iters=num_iters, tol=tol)
    # 对应原矩阵的特征值为 shift + 1/val
    if abs(val) < 1e-30:
        raise np.linalg.LinAlgError("inverse power failed (small val)")
    eig = shift + 1.0 / val
    return eig, vec

# -------------------------
# 共轭梯度法（对称正定矩阵求解 Ax=b）
# -------------------------

def conjugate_gradient(A: Iterable, b: Iterable, x0: Optional[Iterable] = None, tol: float = 1e-10, max_iter: Optional[int] = None) -> np.ndarray:
    """
    共轭梯度法求解 Ax = b，A 应为对称正定（或近似）。
    返回近似解 x。
    """
    A = _to_np_mat(A)
    b = _to_np_vec(b)
    n, m = A.shape
    if n != m:
        raise ValueError("CG requires square matrix")
    if b.size != n:
        raise ValueError("b size mismatch")
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = _to_np_vec(x0).astype(float)
    r = b - A.dot(x)
    p = r.copy()
    rs_old = np.dot(r, r)
    if max_iter is None:
        max_iter = n
    for i in range(max_iter):
        Ap = A.dot(p)
        alpha = rs_old / np.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = np.dot(r, r)
        if math.sqrt(rs_new) < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x

# -------------------------
# 额外工具：矩阵范数、Frobenius、条件数
# -------------------------

def matrix_norm(A: Iterable, ord: Optional[str] = 'fro') -> float:
    A = _to_np_mat(A)
    return float(np.linalg.norm(A, ord=ord))

def condition_number(A: Iterable) -> float:
    A = _to_np_mat(A)
    return float(np.linalg.cond(A))

# -------------------------
# 简单测试 / 示例（作为脚本运行）
# -------------------------
if __name__ == "__main__":
    # 向量示例
    a = [1, 2, 3]; b = [4, 5, 6]
    print("dot:", dot(a, b))
    print("l2 dist:", l2_distance(a, b))
    print("cosine:", cosine_similarity(a, b))

    # 矩阵示例
    A = np.array([[4.0, 2.0, 0.6],
                  [2.0, 5.0, 1.0],
                  [0.6, 1.0, 3.0]])
    b = np.array([1.0, 2.0, 3.0])
    print("A symmetric:", mat_is_symmetric(A))
    print("solve_gauss:", solve_gauss(A, b))
    L, U, P = lu_decomposition(A, pivot=True)
    print("LU done, check L@U:", np.allclose(L.dot(U), P.dot(A)))
    print("cholesky L:", cholesky_decomposition(A))
    Q, R = qr_householder(A)
    print("QR check:", np.allclose(Q.dot(R), A))
    U, s, Vt = svd_decompose(A)
    print("svd singulars:", s)
    print("pseudo inv check:", np.allclose(pseudo_inverse(A).dot(A).dot(pseudo_inverse(A)), pseudo_inverse(A)))
    eig, vec = power_method(A)
    print("power method eigenvalue:", eig)
    x_cg = conjugate_gradient(A, b)
    print("cg solution close to np.linalg.solve:", np.allclose(x_cg, np.linalg.solve(A, b), atol=1e-6))
