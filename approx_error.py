"""
Thí nghiệm mở rộng 2 (thay cho phần 2 của benchmark_scaling.py):
sai số của phép xấp xỉ ma trận kề bằng đặc trưng ngẫu nhiên (Eq. 17 của BigST)
theo số chiều đặc trưng ngẫu nhiên r và theo độ "sắc" của ma trận kề.

  A     = softmax(E1 E2^T / (tau*sqrt(d)))                    (ma trận kề thật, N x N)
  A_hat = D^-1 phi(E1) phi(E2)^T  (chuẩn hóa hàng)             (xấp xỉ của BigST)
  Sai số = khoảng cách biến phân toàn phần trung bình theo hàng
         = trung bình_i  0.5 * sum_j |A_hat[i,j] - A[i,j]|     (0 = trùng khớp, 1 = khác hoàn toàn)

Độ sắc của A được điều khiển bằng độ lệch chuẩn s của embedding nút: s càng lớn thì điểm
attention càng chênh lệch (một vài nút lân cận chiếm gần hết trọng số).
Dùng đúng các hàm create_random_matrix, random_feature_map trong model.py.

Ví dụ:
    python approx_error.py
"""
import argparse
import csv
import math
import os

import torch

from model import create_random_matrix, random_feature_map

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--N', type=int, default=1000)
parser.add_argument('--node_dim', type=int, default=32)
parser.add_argument('--tau', type=float, default=0.25)
parser.add_argument('--seeds', type=int, default=5)
args = parser.parse_args()
dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
os.makedirs('results', exist_ok=True)
N, d, tau = args.N, args.node_dim, args.tau
R_LIST = [8, 16, 32, 64, 128, 256]
S_LIST = [0.125, 0.25, 0.5]

rows = []
with torch.no_grad():
    for s_emb in S_LIST:
        g = torch.Generator().manual_seed(0)
        E1 = (torch.randn(1, N, 1, d, generator=g) * s_emb).to(dev)
        E2 = (torch.randn(1, N, 1, d, generator=g) * s_emb).to(dev)
        q, k = E1 / math.sqrt(tau), E2 / math.sqrt(tau)
        scores = torch.einsum('bnhd,bmhd->nm', q, k) / math.sqrt(d)
        A = torch.softmax(scores, dim=-1)                                   # (N, N)
        logit_std = scores.std().item()
        top = A.max(dim=-1).values.mean().item()                            # trọng số lớn nhất TB mỗi hàng
        for r in R_LIST:
            errs = []
            for seed in range(args.seeds):
                P = create_random_matrix(r, d, seed=100 + seed).to(dev)
                fq = random_feature_map(q, True, P)[0, :, 0, :]             # (N, r)
                fk = random_feature_map(k, False, P)[0, :, 0, :]            # (N, r)
                A_hat = fq @ fk.T
                A_hat = A_hat / A_hat.sum(dim=-1, keepdim=True)
                errs.append((0.5 * (A_hat - A).abs().sum(dim=-1)).mean().item())
            mean = sum(errs) / len(errs)
            std = (sum((e - mean) ** 2 for e in errs) / len(errs)) ** 0.5
            rows.append([s_emb, f'{logit_std:.3f}', f'{top:.4f}', r, f'{mean:.4f}', f'{std:.4f}'])
            print(f's={s_emb:<5} (độ lệch chuẩn điểm attention {logit_std:.2f}) | r={r:4d}: '
                  f'sai số {mean:.4f} ± {std:.4f}', flush=True)

with open('results/approx_error.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['emb_std', 'logit_std', 'mean_max_weight', 'r', 'tv_error_mean', 'tv_error_std'])
    w.writerows(rows)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5.5, 4))
    for s_emb in S_LIST:
        sub = [r for r in rows if r[0] == s_emb]
        plt.errorbar([r[3] for r in sub], [float(r[4]) for r in sub], yerr=[float(r[5]) for r in sub],
                     marker='o', capsize=3, label=f'độ lệch chuẩn điểm = {sub[0][1]}')
    plt.xscale('log', base=2)
    plt.xlabel('Số chiều đặc trưng ngẫu nhiên r')
    plt.ylabel('Sai số xấp xỉ ma trận kề (TV)')
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig('results/approx_error.png', dpi=200)
    print('Đã lưu hình: results/approx_error.png')
except ImportError:
    print('Chưa có matplotlib nên chưa vẽ hình')
print('Đã lưu: results/approx_error.csv')