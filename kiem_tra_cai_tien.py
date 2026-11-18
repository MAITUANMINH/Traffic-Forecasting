import argparse
import time

import numpy as np
import torch

import util
from model import BigST
from model_cai_tien import BigSTCaiTien

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
N, B = 1000, 4


def tao_args(**kw):
    a = dict(tau=0.25, num_layers=3, random_feature_dim=64, use_residual=True, use_bn=True,
             use_spatial=False, use_long=True, dropout=0.3, time_num=288, week_num=7,
             num_nodes=N, node_dim=32, time_dim=32, hid_dim=32, output_length=12, in_dim=3,
             local_graph=False, predict_delta=False, local_sparse=False)
    a.update(kw)
    return argparse.Namespace(**a)


adj = util.load_adj('data/pems_sub1000/adj_pems.npy')
supports = [torch.tensor(i).to(dev) for i in adj]
edge_indices = torch.nonzero(supports[0] > 0)
print(f'Thiết bị: {dev}; số cạnh: {len(edge_indices)}')

torch.manual_seed(0)
x = torch.randn(B, N, 12, 3, device=dev)
x[..., 1] = torch.randint(0, 288, (B, N, 1), device=dev).float() / 288   # thời điểm trong ngày
x[..., 2] = torch.randint(0, 7, (B, N, 1), device=dev).float()           # ngày trong tuần
feat = torch.randn(B, N, 44, device=dev)

# 1. tắt cải tiến -> giống bản gốc
torch.manual_seed(1); m0 = BigST(tao_args(), supports, edge_indices).to(dev).eval()
torch.manual_seed(1); m1 = BigSTCaiTien(tao_args(), supports, edge_indices).to(dev).eval()
with torch.no_grad():
    y0, _ = m0(x, feat)
    y1, _ = m1(x, feat)
print(f'1. Tắt cải tiến, chênh lệch so với bản gốc: {(y0 - y1).abs().max().item():.2e} (phải ~0)')

# 2. các cấu hình
for ten, kw in [('A', dict(local_graph=True)), ('B', dict(predict_delta=True)),
                ('A+B', dict(local_graph=True, predict_delta=True))]:
    m = BigSTCaiTien(tao_args(**kw), supports, edge_indices).to(dev).train()
    t = time.time()
    y, _ = m(x, feat)
    y.mean().backward()
    so_tham_so = sum(p.numel() for p in m.parameters())
    print(f'2. {ten:4s}: đầu ra {tuple(y.shape)} (phải ({B}, {N}, 12)), '
          f'tham số {so_tham_so:,} (gốc {sum(p.numel() for p in m0.parameters()):,}), '
          f'{(time.time() - t) * 1000:.0f} ms')

# 3. dày vs thưa
torch.manual_seed(2); md = BigSTCaiTien(tao_args(local_graph=True), supports, edge_indices).to(dev).eval()
torch.manual_seed(2); ms = BigSTCaiTien(tao_args(local_graph=True, local_sparse=True), supports, edge_indices).to(dev).eval()
with torch.no_grad():
    print(f'3. Dày so với thưa, chênh lệch: {(md(x, feat)[0] - ms(x, feat)[0]).abs().max().item():.2e} (phải ~0)')
print('Xong, không có lỗi.')