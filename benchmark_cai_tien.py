"""
Cuối kỳ: chi phí của nhánh đồ thị cục bộ (cải tiến A) theo số nút N.

So sánh 3 cách lan truyền không gian cho 1 lớp, đo thời gian (xuôi + ngược) và bộ nhớ GPU đỉnh:
  * Dense      : ma trận kề đầy đủ N x N                       -> O(N^2)
  * BigST      : tích chập tuyến tính hóa (đặc trưng ngẫu nhiên) -> O(N)
  * BigST + A  : BigST + A_hat_road · H · W_loc (A_hat thưa)     -> O(N + E) = O(N)
Ma trận kề đường được mô phỏng với ~7 cạnh/nút (tập con 1.000 nút thật có bậc TB 6,66).

    python -X utf8 benchmark_cai_tien.py
Kết quả: results/scaling_cai_tien.csv, results/scaling_cai_tien.png
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
parser.add_argument('--degree', type=int, default=7, help='số cạnh mỗi nút của đồ thị đường mô phỏng')
parser.add_argument('--node_dim', type=int, default=32)
parser.add_argument('--hid_dim', type=int, default=128)
parser.add_argument('--r', type=int, default=64)
parser.add_argument('--tau', type=float, default=0.25)
parser.add_argument('--repeats', type=int, default=10)
args = parser.parse_args()
dev = torch.device(args.device)
os.makedirs('results', exist_ok=True)
d, C, tau = args.node_dim, args.hid_dim, args.tau


def road_graph(N, k, seed=0):
    """Đồ thị thưa giống mạng đường: mỗi nút nối với k nút có chỉ số gần nó, chuẩn hóa D^-1 A."""
    g = torch.Generator().manual_seed(seed)
    src = torch.arange(N).repeat_interleave(k)
    off = torch.randint(1, 20, (N * k,), generator=g) * (torch.randint(0, 2, (N * k,), generator=g) * 2 - 1)
    dst = (src + off) % N
    idx = torch.stack([src, dst])
    val = torch.ones(N * k)
    A = torch.sparse_coo_tensor(idx, val, (N, N)).coalesce()
    deg = torch.sparse.sum(A, dim=1).to_dense()
    i = A.indices()
    A = torch.sparse_coo_tensor(i, A.values() / deg[i[0]], (N, N)).coalesce()
    return A.to(dev)


def dense_conv(E1, E2, H):
    q, k = E1 / math.sqrt(tau), E2 / math.sqrt(tau)
    A = torch.softmax(torch.einsum('bnhd,bmhd->bhnm', q, k) / math.sqrt(d), dim=-1)
    return torch.einsum('bhnm,bmhc->bnhc', A, H)


def linear_conv(E1, E2, H, P):
    q = random_feature_map(E1 / math.sqrt(tau), True, P)
    k = random_feature_map(E2 / math.sqrt(tau), False, P)
    return linear_kernel(H, q, k)


def local_conv(H, A, W):
    """A_hat · H · W_loc với A_hat thưa. H: (1, N, 1, C)."""
    h = H[0, :, 0, :] @ W                     # (N, C)
    return torch.sparse.mm(A, h)[None, :, None, :]


def bench(kind, N):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)
    try:
        g = torch.Generator().manual_seed(0)
        E1 = torch.randn(1, N, 1, d, generator=g).to(dev).requires_grad_()
        E2 = torch.randn(1, N, 1, d, generator=g).to(dev).requires_grad_()
        H = torch.randn(1, N, 1, C, generator=g).to(dev).requires_grad_()
        P = create_random_matrix(args.r, d, seed=0).to(dev)
        if kind == 'dense':
            fn = lambda: dense_conv(E1, E2, H)
        elif kind == 'linear':
            fn = lambda: linear_conv(E1, E2, H, P)
        else:
            A = road_graph(N, args.degree)
            W = (torch.randn(C, C, generator=g) / math.sqrt(C)).to(dev).requires_grad_()
            fn = lambda: linear_conv(E1, E2, H, P) + local_conv(H, A, W)
        for _ in range(2):
            fn().sum().backward()
        torch.cuda.synchronize(dev)
        t0 = time.time()
        for _ in range(args.repeats):
            fn().sum().backward()
        torch.cuda.synchronize(dev)
        ms = (time.time() - t0) / args.repeats * 1000
        return f'{ms:.2f}', f'{torch.cuda.max_memory_allocated(dev) / 2 ** 20:.1f}'
    except RuntimeError as e:
        if 'out of memory' not in str(e).lower():
            raise
        return 'OOM', 'OOM'
    finally:
        torch.cuda.empty_cache()


print(f'GPU: {torch.cuda.get_device_name(dev)}')
rows = []
for N in args.nodes:
    r = [N]
    for kind in ['dense', 'linear', 'local']:
        r += list(bench(kind, N))
    rows.append(r)
    print(f'N={N:6d} | dense {r[1]:>8} ms {r[2]:>8} MB | BigST {r[3]:>7} ms {r[4]:>7} MB | '
          f'BigST+A {r[5]:>7} ms {r[6]:>7} MB', flush=True)

with open('results/scaling_cai_tien.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['N', 'dense_ms', 'dense_MB', 'bigst_ms', 'bigst_MB', 'bigst_A_ms', 'bigst_A_MB'])
    w.writerows(rows)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
names = [('Ma trận kề đầy đủ', 1, 'tab:red'), ('BigST', 3, 'tab:blue'), ('BigST + nhánh cục bộ (A)', 5, 'tab:green')]
for ax, extra, label in [(axes[0], 0, 'Thời gian xuôi + ngược (ms)'), (axes[1], 1, 'Bộ nhớ GPU đỉnh (MB)')]:
    for name, col, color in names:
        xs = [r[0] for r in rows if r[col + extra] != 'OOM']
        ys = [float(r[col + extra]) for r in rows if r[col + extra] != 'OOM']
        ax.plot(xs, ys, marker='o', label=name, color=color)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Số nút N'); ax.set_ylabel(label); ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('results/scaling_cai_tien.png', dpi=200)
print('Đã lưu: results/scaling_cai_tien.csv, results/scaling_cai_tien.png')
