"""
Thí nghiệm mở rộng: kiểm chứng độ phức tạp tuyến tính của phép tích chập không gian
trong BigST (mục 3.3.2, Eq. 15-19) so với phép tích chập dùng ma trận kề đầy đủ (dense).

Thí nghiệm 1 - Khả năng mở rộng theo số nút N:
  * Dense    : A = softmax(E1 E2^T / (tau*sqrt(d)))  (ma trận N x N),  Y = A H      -> O(N^2)
  * Tuyến tính: Y = D^-1 phi(E1) (phi(E2)^T H)   (không tạo ma trận N x N)         -> O(N)
  Đo thời gian (lan truyền xuôi + ngược) và bộ nhớ GPU đỉnh cho từng N.

Thí nghiệm 2 - Sai số xấp xỉ theo số chiều đặc trưng ngẫu nhiên r (Eq. 17):
  sai số tương đối ||Y_tuyến_tính - Y_dense|| / ||Y_dense|| với N = 1000.

Dùng lại đúng các hàm của BigST trong model.py: create_random_matrix, random_feature_map,
linear_kernel. Kết quả: results/scaling.csv, results/approx_error.csv (+ hình nếu có matplotlib).

Ví dụ:
    python benchmark_scaling.py
"""
import argparse
import csv
import math
import os
import time

import torch

from model import create_random_matrix, random_feature_map, linear_kernel

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--nodes', type=int, nargs='+', default=[1000, 2000, 5000, 10000, 20000, 50000, 100000])
parser.add_argument('--node_dim', type=int, default=32, help='số chiều embedding nút d (W_1, W_2 trong code)')
parser.add_argument('--hid_dim', type=int, default=128, help='số chiều đặc trưng nút (hid_dim*4 trong code)')
parser.add_argument('--r', type=int, default=64, help='số chiều đặc trưng ngẫu nhiên')
parser.add_argument('--tau', type=float, default=0.25)
parser.add_argument('--repeats', type=int, default=10)
args = parser.parse_args()
dev = torch.device(args.device)
os.makedirs('results', exist_ok=True)
d, C, tau = args.node_dim, args.hid_dim, args.tau


def dense_conv(E1, E2, H):
    """Tích chập với ma trận kề đầy đủ N x N (attention softmax thông thường)."""
    q, k = E1 / math.sqrt(tau), E2 / math.sqrt(tau)
    scores = torch.einsum('bnhd,bmhd->bhnm', q, k) / math.sqrt(d)      # (B, 1, N, N)
    A = torch.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bmhc->bnhc', A, H)                        # (B, N, 1, C)


def linear_conv(E1, E2, H, P):
    """Tích chập tuyến tính của BigST: phi(.) là đặc trưng ngẫu nhiên dương (Eq. 5, 17-19)."""
    q = random_feature_map(E1 / math.sqrt(tau), True, P)               # (B, N, 1, r)
    k = random_feature_map(E2 / math.sqrt(tau), False, P)
    return linear_kernel(H, q, k)                                       # (B, N, 1, C)


def make_inputs(N, seed=0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    E1 = torch.randn(1, N, 1, d, generator=g).to(dev).requires_grad_()
    E2 = torch.randn(1, N, 1, d, generator=g).to(dev).requires_grad_()
    H = torch.randn(1, N, 1, C, generator=g).to(dev).requires_grad_()
    return E1, E2, H


def bench(kind, N):
    """Trả về (ms mỗi lần xuôi+ngược, MB bộ nhớ đỉnh) hoặc ('OOM', 'OOM')."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)
    try:
        E1, E2, H = make_inputs(N)
        P = create_random_matrix(args.r, d, seed=0).to(dev)
        fn = (lambda: dense_conv(E1, E2, H)) if kind == 'dense' else (lambda: linear_conv(E1, E2, H, P))
        for _ in range(2):                                              # khởi động GPU
            fn().sum().backward()
        torch.cuda.synchronize(dev)
        t0 = time.time()
        for _ in range(args.repeats):
            fn().sum().backward()
        torch.cuda.synchronize(dev)
        ms = (time.time() - t0) / args.repeats * 1000
        mem = torch.cuda.max_memory_allocated(dev) / 2 ** 20
        return f'{ms:.2f}', f'{mem:.1f}'
    except RuntimeError as e:          # torch.cuda.OutOfMemoryError cũng là RuntimeError
        if 'out of memory' not in str(e).lower():
            raise
        return 'OOM', 'OOM'
    finally:
        torch.cuda.empty_cache()


# ---------------- Thí nghiệm 1 ----------------
print(f'GPU: {torch.cuda.get_device_name(dev)}')
print('Thí nghiệm 1: thời gian và bộ nhớ theo số nút N')
rows = []
for N in args.nodes:
    dm, dmem = bench('dense', N)
    lm, lmem = bench('linear', N)
    rows.append([N, dm, dmem, lm, lmem])
    print(f'N={N:6d} | dense: {dm:>8} ms, {dmem:>8} MB | tuyến tính: {lm:>8} ms, {lmem:>8} MB', flush=True)
with open('results/scaling.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['N', 'dense_ms', 'dense_MB', 'linear_ms', 'linear_MB'])
    w.writerows(rows)

# ---------------- Thí nghiệm 2 ----------------
print('\nThí nghiệm 2: sai số xấp xỉ theo r (N = 1000, trung bình 5 lần)')
err_rows = []
with torch.no_grad():
    for r in [8, 16, 32, 64, 128, 256]:
        errs = []
        for s in range(5):
            E1, E2, H = make_inputs(1000, seed=s)
            P = create_random_matrix(r, d, seed=100 + s).to(dev)
            Yd, Yl = dense_conv(E1, E2, H), linear_conv(E1, E2, H, P)
            errs.append((torch.norm(Yl - Yd) / torch.norm(Yd)).item())
        mean, std = sum(errs) / len(errs), (sum((e - sum(errs) / len(errs)) ** 2 for e in errs) / len(errs)) ** 0.5
        err_rows.append([r, f'{mean:.4f}', f'{std:.4f}'])
        print(f'r={r:4d}: sai số tương đối {mean:.4f} ± {std:.4f}', flush=True)
with open('results/approx_error.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['r', 'rel_error_mean', 'rel_error_std'])
    w.writerows(err_rows)

# ---------------- Vẽ hình (nếu có matplotlib) ----------------
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for col, ax, label in [(1, axes[0], 'Thời gian xuôi + ngược (ms)'), (2, axes[1], 'Bộ nhớ GPU đỉnh (MB)')]:
        for kind, off, name in [('dense', 0, 'Ma trận kề đầy đủ'), ('linear', 2, 'Tuyến tính (BigST)')]:
            xs = [r[0] for r in rows if r[col + off] != 'OOM']
            ys = [float(r[col + off]) for r in rows if r[col + off] != 'OOM']
            ax.plot(xs, ys, marker='o', label=name)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('Số nút N'); ax.set_ylabel(label); ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout(); plt.savefig('results/scaling.png', dpi=200)

    plt.figure(figsize=(5, 4))
    plt.errorbar([r[0] for r in err_rows], [float(r[1]) for r in err_rows],
                 yerr=[float(r[2]) for r in err_rows], marker='o', capsize=3)
    plt.xscale('log', base=2); plt.xlabel('Số chiều đặc trưng ngẫu nhiên r')
    plt.ylabel('Sai số tương đối'); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig('results/approx_error.png', dpi=200)
    print('\nĐã lưu hình: results/scaling.png, results/approx_error.png')
except ImportError:
    print('\nChưa có matplotlib nên chưa vẽ hình (cài bằng: python -m pip install matplotlib)')
print('Đã lưu: results/scaling.csv, results/approx_error.csv')