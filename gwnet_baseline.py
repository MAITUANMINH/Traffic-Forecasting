"""
Baseline Graph WaveNet (GWNET; Wu và cộng sự, IJCAI 2019) - mô hình cơ sở mạnh nhất trong Bảng 2
của bài báo BigST. GWNET học một ma trận kề thích nghi DÀY  A_adp = softmax(ReLU(E1 E2)) kích
thước N x N, kết hợp tích chập nhân quả giãn (dilated causal conv) theo thời gian.

Cấu trúc và siêu tham số theo code gốc (github.com/nnzhan/Graph-WaveNet): 4 khối x 2 lớp, kernel 2,
kênh residual/dilation 32, skip 256, end 512, khuếch tán bậc 2 với ma trận chuyển tiếp xuôi/ngược
+ ma trận thích nghi, Adam lr 0.001, weight decay 1e-4, dropout 0.3, cắt gradient 5.
Batch mặc định 32 (code gốc 64) để vừa GPU 8GB; nếu báo hết bộ nhớ, dùng --batch_size 16.

Để so sánh công bằng với BigST (run_fast.py): cùng dữ liệu, cùng chuẩn hóa, cùng tập mẫu
(--min_t 2015), cùng hàm đo (metrics.py), cùng cách lấy trung bình theo batch 16 khi kiểm tra.
Đầu vào: 12 bước gần nhất, 2 kênh (tốc độ, thời điểm trong ngày) như code gốc GWNET.

Ví dụ:
    python gwnet_baseline.py --epochs 50
"""
import argparse
import copy
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import metrics
import util

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--data_path', type=str, default='data/pems_sub1000/short_term')
parser.add_argument('--adjdata', type=str, default='data/pems_sub1000/adj_pems.npy')
parser.add_argument('--name', type=str, default='GWNET')
parser.add_argument('--min_t', type=int, default=2015)
parser.add_argument('--in_dim', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=32, help='code gốc dùng 64; giảm còn 32 (hoặc 16) cho GPU 8GB')
parser.add_argument('--eval_batch_size', type=int, default=16)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--patience', type=int, default=30)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
dev = torch.device(args.device)
os.makedirs('results', exist_ok=True)
os.makedirs('checkpoint', exist_ok=True)
L_IN, L_OUT = 12, 12


# ======================= Mô hình Graph WaveNet =======================
class NConv(nn.Module):
    def forward(self, x, A):                      # x: (B, C, N, T), A: (N, N)
        return torch.einsum('ncvl,vw->ncwl', x, A).contiguous()


class GCN(nn.Module):
    """Tích chập đồ thị khuếch tán bậc `order` trên nhiều ma trận hỗ trợ."""
    def __init__(self, c_in, c_out, dropout, support_len, order=2):
        super().__init__()
        self.nconv = NConv()
        self.mlp = nn.Conv2d((order * support_len + 1) * c_in, c_out, kernel_size=(1, 1))
        self.dropout, self.order = dropout, order

    def forward(self, x, supports):
        out = [x]
        for a in supports:
            x1 = self.nconv(x, a)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x1 = self.nconv(x1, a)
                out.append(x1)
        h = self.mlp(torch.cat(out, dim=1))
        return F.dropout(h, self.dropout, training=self.training)


class GWNet(nn.Module):
    def __init__(self, num_nodes, supports, dropout=0.3, in_dim=2, out_dim=12, residual=32, dilation=32,
                 skip=256, end=512, kernel_size=2, blocks=4, layers=2):
        super().__init__()
        self.supports = supports
        self.blocks, self.layers, self.dropout = blocks, layers, dropout
        self.start_conv = nn.Conv2d(in_dim, residual, kernel_size=(1, 1))
        # ma trận kề thích nghi: A_adp = softmax(ReLU(E1 E2)), mỗi nút một vector 10 chiều
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10))
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes))
        support_len = len(supports) + 1
        self.filter_convs, self.gate_convs = nn.ModuleList(), nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn, self.gconv = nn.ModuleList(), nn.ModuleList()
        receptive_field = 1
        for _ in range(blocks):
            scope, d = kernel_size - 1, 1
            for _ in range(layers):
                self.filter_convs.append(nn.Conv2d(residual, dilation, (1, kernel_size), dilation=d))
                self.gate_convs.append(nn.Conv2d(residual, dilation, (1, kernel_size), dilation=d))
                self.skip_convs.append(nn.Conv2d(dilation, skip, (1, 1)))
                self.bn.append(nn.BatchNorm2d(residual))
                self.gconv.append(GCN(dilation, residual, dropout, support_len))
                d *= 2
                receptive_field += scope
                scope *= 2
        self.receptive_field = receptive_field     # = 13 với cấu hình mặc định
        self.end_conv_1 = nn.Conv2d(skip, end, (1, 1))
        self.end_conv_2 = nn.Conv2d(end, out_dim, (1, 1))

    def forward(self, x):                          # x: (B, in_dim, N, T)
        if x.size(3) < self.receptive_field:
            x = F.pad(x, (self.receptive_field - x.size(3), 0, 0, 0))
        x = self.start_conv(x)
        skip = 0
        adp = F.softmax(F.relu(self.nodevec1 @ self.nodevec2), dim=1)   # (N, N) - chi phí O(N^2)
        supports = self.supports + [adp]
        for i in range(self.blocks * self.layers):
            res = x
            x = torch.tanh(self.filter_convs[i](res)) * torch.sigmoid(self.gate_convs[i](res))
            s = self.skip_convs[i](x)
            skip = s + (skip[:, :, :, -s.size(3):] if torch.is_tensor(skip) else 0)
            x = self.gconv[i](x, supports)
            x = x + res[:, :, :, -x.size(3):]
            x = self.bn[i](x)
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        return self.end_conv_2(x)                  # (B, 12, N, 1)


# ======================= Dữ liệu =======================
data = {s: np.load(os.path.join(args.data_path, f'{s}.npy')).astype(np.float32) for s in ['train', 'val', 'test']}
scaler = util.StandardScaler(mean=data['train'][..., 0].mean(), std=data['train'][..., 0].std())
for s in data:
    data[s][..., 0] = scaler.transform(data[s][..., 0])
N = data['train'].shape[1]
x_off, y_off = np.arange(-(L_IN - 1), 1), np.arange(1, L_OUT + 1)


def batches(split, bs, shuffle):
    T = data[split].shape[0]
    idx = np.arange(max(args.min_t, L_IN - 1), T - L_OUT)
    if shuffle:
        idx = np.random.permutation(idx)
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        x = torch.from_numpy(data[split][b[:, None] + x_off][..., :args.in_dim]).to(dev)   # (B, T, N, C)
        y = torch.from_numpy(data[split][b[:, None] + y_off][..., 0]).to(dev)                # (B, T, N)
        yield x.permute(0, 3, 2, 1), y.transpose(1, 2)                                       # (B,C,N,T), (B,N,T)


# ma trận chuyển tiếp xuôi và ngược (doubletransition trong code gốc)
adj = np.load(args.adjdata).astype(np.float32)
supports = [torch.tensor(np.asarray(util.asym_adj(adj)), dtype=torch.float32, device=dev),
            torch.tensor(np.asarray(util.asym_adj(adj.T)), dtype=torch.float32, device=dev)]

model = GWNet(N, supports, dropout=args.dropout, in_dim=args.in_dim).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
n_params = sum(p.numel() for p in model.parameters())
print(f'Số tham số: {n_params}, receptive field: {model.receptive_field}')


def predict(x):
    out = model(x)[..., 0]                         # (B, 12, N)
    return scaler.inverse_transform(out.transpose(1, 2))        # (B, N, 12), đơn vị gốc


# ======================= Huấn luyện =======================
best_val, best_state, best_epoch, wait = np.inf, None, 0, 0
epoch_times = []
if dev.type == 'cuda':
    torch.cuda.reset_peak_memory_stats(dev)
for ep in range(1, args.epochs + 1):
    model.train()
    t1 = time.time()
    tr = []
    for x, y in batches('train', args.batch_size, True):
        opt.zero_grad()
        pred = predict(x)
        real = scaler.inverse_transform(y)
        loss = metrics.masked_mae(pred, real, 0.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        opt.step()
        tr.append(loss.item())
    t_train = time.time() - t1
    epoch_times.append(t_train)

    model.eval()
    va = []
    with torch.no_grad():
        for x, y in batches('val', args.eval_batch_size, False):
            pred, real = predict(x), scaler.inverse_transform(y)
            va.append([metrics.masked_mae(pred, real, 0.0).item(), metrics.masked_mape(pred, real, 0.0).item(),
                       metrics.masked_rmse(pred, real, 0.0).item()])
    va = np.mean(va, axis=0)
    print(f'Epoch {ep:03d} | train {t_train:.1f}s | Train MAE {np.mean(tr):.4f} | '
          f'Valid MAE {va[0]:.4f}, MAPE {va[1]:.4f}, RMSE {va[2]:.4f}', flush=True)
    if va[0] < best_val:
        best_val, best_epoch, wait = va[0], ep, 0
        best_state = copy.deepcopy(model.state_dict())
    else:
        wait += 1
        if wait >= args.patience:
            print('Dừng sớm (early stopping)')
            break

# ======================= Kiểm tra =======================
model.load_state_dict(best_state)
torch.save(best_state, f'checkpoint/{args.name}.pth')
model.eval()
sums, nb = np.zeros((3, L_OUT)), 0
t1 = time.time()
with torch.no_grad():
    for x, y in batches('test', args.eval_batch_size, False):
        pred, real = predict(x), scaler.inverse_transform(y)
        for h in range(L_OUT):
            sums[:, h] += [metrics.masked_mae(pred[..., h], real[..., h], 0.0).item(),
                           metrics.masked_mape(pred[..., h], real[..., h], 0.0).item(),
                           metrics.masked_rmse(pred[..., h], real[..., h], 0.0).item()]
        nb += 1
test_time = time.time() - t1
res = sums / nb

print(f'\nEpoch tốt nhất: {best_epoch}, Valid MAE {best_val:.4f}')
for h in range(L_OUT):
    print(f'Horizon {h + 1:2d}: Test MAE {res[0, h]:.4f}, MAPE {res[1, h]:.4f}, RMSE {res[2, h]:.4f}')
print(f'Trung bình 12 horizon: Test MAE {res[0].mean():.4f}, MAPE {res[1].mean():.4f}, RMSE {res[2].mean():.4f}')
peak = torch.cuda.max_memory_allocated(dev) / 2 ** 20 if dev.type == 'cuda' else float('nan')
print(f'Thời gian huấn luyện TB: {np.mean(epoch_times):.1f} giây/epoch; test {test_time:.1f} giây; '
      f'bộ nhớ GPU đỉnh {peak:.0f} MB; số tham số {n_params}')

with open(f'results/{args.name}.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['horizon', 'MAE', 'MAPE', 'RMSE'])
    for h in range(L_OUT):
        w.writerow([h + 1, f'{res[0, h]:.4f}', f'{res[1, h]:.4f}', f'{res[2, h]:.4f}'])
    w.writerow(['avg', f'{res[0].mean():.4f}', f'{res[1].mean():.4f}', f'{res[2].mean():.4f}'])
    w.writerow([])
    w.writerow(['best_epoch', best_epoch])
    w.writerow(['train_sec_per_epoch', f'{np.mean(epoch_times):.1f}'])
    w.writerow(['num_params', n_params])
    w.writerow(['peak_gpu_MB', f'{peak:.1f}'])
    w.writerow(['test_sec', f'{test_time:.1f}'])
print(f'Đã lưu kết quả: results/{args.name}.csv')