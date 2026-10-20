
import argparse
import os
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import util
from model import BigST

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--data_path', type=str, default='data/pems_sub1000/short_term')
parser.add_argument('--adjdata', type=str, default='data/pems_sub1000/adj_pems.npy')
parser.add_argument('--feat_path', type=str, default='data/pems_sub1000/features')
parser.add_argument('--bigst_ckpt', type=str, default='checkpoint/bigst_B_BigST.pth')
parser.add_argument('--gwnet_ckpt', type=str, default='checkpoint/GWNET.pth')
parser.add_argument('--node', type=int, default=-1, help='chỉ số nút trong tập con (mặc định: tự chọn)')
parser.add_argument('--start_day', type=int, default=1, help='bắt đầu từ nửa đêm thứ mấy trong tập test')
parser.add_argument('--days', type=int, default=2)
parser.add_argument('--min_t', type=int, default=2015)
args = parser.parse_args()
dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
os.makedirs('results', exist_ok=True)
P, H = 288, 12                                    # số bước/ngày, số bước dự báo


# ============ Graph WaveNet (giống hệt gwnet_baseline.py để nạp được checkpoint) ============
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
train = np.load(os.path.join(args.data_path, 'train.npy')).astype(np.float32)
test = np.load(os.path.join(args.data_path, 'test.npy')).astype(np.float32)
mean, std = train[..., 0].mean(), train[..., 0].std()
del train
raw = test[..., 0].copy()                          # tốc độ gốc (dặm/giờ) để vẽ
test[..., 0] = (test[..., 0] - mean) / std         # chuẩn hóa giống lúc huấn luyện
feats = np.load(os.path.join(args.feat_path, 'test.npy'), mmap_mode='r')
T, N, _ = test.shape

# khoảng thời gian vẽ: bắt đầu từ nửa đêm thứ start_day, đủ lịch sử cho mọi cửa sổ
slot = np.floor(test[:, 0, 1] * P + 1e-6).astype(int) % P
midnights = [t for t in range(args.min_t + H, T) if slot[t] == 0]
s = midnights[min(args.start_day, len(midnights) - 1)]
e = min(s + args.days * P, T)
targets = np.arange(s, e)                           # các thời điểm cần vẽ

node = args.node if args.node >= 0 else int(np.argmax(raw[s:e].std(axis=0)))
node_ids = os.path.join(os.path.dirname(os.path.normpath(args.data_path)), 'node_ids.npy')
orig = int(np.load(node_ids)[node]) if os.path.exists(node_ids) else node
print(f'Vẽ nút {node} (chỉ số trong bộ đầy đủ: {orig}), thời điểm test {s} → {e}')

# ======================= Mô hình =======================
adj_mx = util.load_adj(args.adjdata)
supports = [torch.tensor(np.asarray(a), dtype=torch.float32, device=dev) for a in adj_mx]
cfg = SimpleNamespace(tau=0.25, num_layers=3, random_feature_dim=64, use_residual=True, use_bn=True,
                      use_spatial=False, use_long=True, dropout=0.3, time_num=288, week_num=7,
                      num_nodes=N, node_dim=32, time_dim=32, output_length=H, in_dim=3, hid_dim=32,
                      input_length=12)
bigst = BigST(cfg, supports, torch.nonzero(supports[0] > 0)).to(dev)
bigst.load_state_dict(torch.load(args.bigst_ckpt, map_location=dev))
bigst.eval()

adj = np.load(args.adjdata).astype(np.float32)
gw_sup = [torch.tensor(np.asarray(util.asym_adj(adj)), dtype=torch.float32, device=dev),
          torch.tensor(np.asarray(util.asym_adj(adj.T)), dtype=torch.float32, device=dev)]
gwnet = GWNet(N, gw_sup).to(dev)
gwnet.load_state_dict(torch.load(args.gwnet_ckpt, map_location=dev))
gwnet.eval()


def forecast(ends):
    """Dự báo 12 bước cho các cửa sổ kết thúc tại `ends`; trả về (BigST, GWNET) dạng (B, N, 12), dặm/giờ."""
    out_b, out_g = [], []
    with torch.no_grad():
        for i in range(0, len(ends), 32):
            b = ends[i:i + 32]
            x = torch.from_numpy(test[b[:, None] + np.arange(-11, 1)]).to(dev)       # (B, 12, N, 3)
            f = torch.from_numpy(np.array(feats[b], dtype=np.float32)).to(dev)       # (B, N, 44)
            yb, _ = bigst(x.transpose(1, 2), f)                                       # (B, N, 12)
            yg = gwnet(x[..., :2].permute(0, 3, 2, 1))[..., 0].transpose(1, 2)       # (B, N, 12)
            out_b.append((yb * std + mean).cpu().numpy())
            out_g.append((yg * std + mean).cpu().numpy())
    return np.concatenate(out_b), np.concatenate(out_g)


# ======================= Vẽ =======================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

hours = (targets - s) / 12.0                         # trục hoành: giờ kể từ nửa đêm đầu tiên
truth = raw[targets, node]
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
for ax, h in zip(axes, [3, 12]):
    ends = targets - h                               # dự báo cho thời điểm τ được tạo tại τ - h
    pb, pg = forecast(ends)
    yb, yg = pb[:, node, h - 1], pg[:, node, h - 1]
    yp = raw[ends, node]                             # Persistence: giữ nguyên giá trị tại τ - h
    mae = lambda y: np.abs(y - truth).mean()
    ax.plot(hours, truth, color='black', lw=1.6, label='Thực tế')
    ax.plot(hours, yp, color='tab:gray', lw=1, ls=':', label=f'Persistence (MAE {mae(yp):.2f})')
    ax.plot(hours, yg, color='tab:orange', lw=1.1, label=f'GWNET (MAE {mae(yg):.2f})')
    ax.plot(hours, yb, color='tab:blue', lw=1.1, label=f'BigST (MAE {mae(yb):.2f})')
    ax.set_title(f'Dự báo trước {h * 5} phút (horizon {h})')
    ax.set_ylabel('Tốc độ (dặm/giờ)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', fontsize=9)
    print(f'h={h:2d}: MAE tại nút {node} — BigST {mae(yb):.3f}, GWNET {mae(yg):.3f}, Persistence {mae(yp):.3f}')
axes[-1].set_xlabel('Giờ (tính từ 0h ngày đầu tiên)')
axes[-1].set_xticks(np.arange(0, hours[-1] + 1, 6))
fig.suptitle(f'Tốc độ thực tế và dự báo tại cảm biến {orig} (tập test, {args.days} ngày)')
plt.tight_layout()
out = f'results/du_bao_nut_{orig}.png'
plt.savefig(out, dpi=200)
print(f'Đã lưu hình: {out}')