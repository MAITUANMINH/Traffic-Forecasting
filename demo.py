
import argparse
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import util
from model import BigST
from model_cai_tien import BigSTCaiTien

parser = argparse.ArgumentParser()
parser.add_argument('--sensor', type=int, default=6072, help='mã cảm biến trong bộ California đầy đủ')
parser.add_argument('--day', type=int, default=1, help='ngày thứ mấy của tập test (tính từ 0)')
parser.add_argument('--time', type=str, default='auto',
                    help='giờ "bây giờ" dạng HH:MM, hoặc auto = lúc tốc độ sắp giảm mạnh nhất trong ngày')
parser.add_argument('--no_gwnet', action='store_true', help='chỉ chạy BigST')
parser.add_argument('--no_open', action='store_true', help='không tự mở hình sau khi vẽ')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--data_path', type=str, default='data/pems_sub1000/short_term')
parser.add_argument('--adjdata', type=str, default='data/pems_sub1000/adj_pems.npy')
parser.add_argument('--feat_path', type=str, default='data/pems_sub1000/features')
parser.add_argument('--bigst_ckpt', type=str, default='checkpoint/bigst_B_BigST.pth')
parser.add_argument('--gwnet_ckpt', type=str, default='checkpoint/GWNET.pth')
parser.add_argument('--cai_tien_ckpt', type=str, default='checkpoint/bigst_C_AB.pth',
                    help='checkpoint BigST cải tiến (A+B); bỏ qua nếu không có file')
parser.add_argument('--min_t', type=int, default=2015)
args = parser.parse_args()

dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
P, H = 288, 12                                     # số bước 5 phút trong ngày, số bước dự báo
os.makedirs('results', exist_ok=True)


def line(c='='):
    print(c * 72)


# ============ Graph WaveNet (giống hệt gwnet_baseline.py để nạp được checkpoint) ============
class NConv(nn.Module):
    def forward(self, x, A):
        return torch.einsum('ncvl,vw->ncwl', x, A).contiguous()


class GCN(nn.Module):
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
        self.receptive_field = receptive_field
        self.end_conv_1 = nn.Conv2d(skip, end, (1, 1))
        self.end_conv_2 = nn.Conv2d(end, out_dim, (1, 1))

    def forward(self, x):
        if x.size(3) < self.receptive_field:
            x = F.pad(x, (self.receptive_field - x.size(3), 0, 0, 0))
        x = self.start_conv(x)
        skip = 0
        adp = F.softmax(F.relu(self.nodevec1 @ self.nodevec2), dim=1)
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
        return self.end_conv_2(x)


# ============================== 1. Dữ liệu ==============================
line()
print(' DEMO DỰ BÁO GIAO THÔNG BẰNG BigST (tập con 1.000 cảm biến, California PeMS)')
line()
t0 = time.time()
train = np.load(os.path.join(args.data_path, 'train.npy'), mmap_mode='r')
mean, std = float(train[..., 0].mean()), float(train[..., 0].std())
del train
test = np.load(os.path.join(args.data_path, 'test.npy')).astype(np.float32)
raw = test[..., 0].copy()                          # tốc độ gốc (dặm/giờ)
test[..., 0] = (test[..., 0] - mean) / std
feats = np.load(os.path.join(args.feat_path, 'test.npy'), mmap_mode='r')   # đặc trưng LSFE tính sẵn
T, N, _ = test.shape

node_ids_path = os.path.join(os.path.dirname(os.path.normpath(args.data_path)), 'node_ids.npy')
node_ids = np.load(node_ids_path) if os.path.exists(node_ids_path) else np.arange(N)
hit = np.where(node_ids == args.sensor)[0]
if len(hit) == 0:
    sys.exit(f'Cảm biến {args.sensor} không nằm trong tập con 1.000 nút. '
             f'Ví dụ các mã hợp lệ: {", ".join(map(str, node_ids[:8]))}, ...')
node = int(hit[0])

# chọn thời điểm "bây giờ"
slot = np.floor(test[:, 0, 1] * P + 1e-6).astype(int) % P
midnights = [t for t in range(args.min_t + H, T) if slot[t] == 0]
day = min(max(args.day, 0), len(midnights) - 1)
m0 = midnights[day]
if args.time == 'auto':
    # lúc tốc độ trong 60 phút tới giảm mạnh nhất so với hiện tại (bắt đầu ùn tắc), trong khung 5h-20h
    cand = np.arange(m0 + 5 * 12, min(m0 + 20 * 12, T - H))
    drop = [raw[t, node] - raw[t + 1:t + 1 + H, node].min() for t in cand]
    now = int(cand[int(np.argmax(drop))])
else:
    hh, mm = map(int, args.time.split(':'))
    now = m0 + hh * 12 + mm // 5
now = min(max(now, args.min_t), T - H - 1)
clock = lambda t: f'{(slot[t] // 12):02d}:{(slot[t] % 12) * 5:02d}'
print(f'Cảm biến       : {args.sensor} (nút số {node} trong tập con)')
print(f'Thời điểm "bây giờ": {clock(now)}, ngày thứ {day} của tập test (mã ngày trong tuần {int(test[now, 0, 2])})')
print(f'Đầu vào        : 12 bước (1 giờ) vừa qua của cả {N} cảm biến + đặc trưng dài hạn 1 tuần (tính sẵn)')
print(f'Đã đọc dữ liệu : {time.time() - t0:.1f} giây, thiết bị {dev}')

# ============================== 2. Mô hình ==============================
adj_mx = util.load_adj(args.adjdata)
supports = [torch.tensor(np.asarray(a), dtype=torch.float32, device=dev) for a in adj_mx]
cfg = SimpleNamespace(tau=0.25, num_layers=3, random_feature_dim=64, use_residual=True, use_bn=True,
                      use_spatial=False, use_long=True, dropout=0.3, time_num=288, week_num=7,
                      num_nodes=N, node_dim=32, time_dim=32, output_length=H, in_dim=3, hid_dim=32,
                      input_length=12)
bigst = BigST(cfg, supports, torch.nonzero(supports[0] > 0)).to(dev)
bigst.load_state_dict(torch.load(args.bigst_ckpt, map_location=dev))
bigst.eval()

cai_tien = None
if os.path.exists(args.cai_tien_ckpt):
    cfg_ct = SimpleNamespace(**vars(cfg), local_graph=True, predict_delta=True, local_sparse=False)
    cai_tien = BigSTCaiTien(cfg_ct, supports, torch.nonzero(supports[0] > 0)).to(dev)
    cai_tien.load_state_dict(torch.load(args.cai_tien_ckpt, map_location=dev))
    cai_tien.eval()

gwnet = None
if not args.no_gwnet:
    adj = np.load(args.adjdata).astype(np.float32)
    gw_sup = [torch.tensor(np.asarray(util.asym_adj(adj)), dtype=torch.float32, device=dev),
              torch.tensor(np.asarray(util.asym_adj(adj.T)), dtype=torch.float32, device=dev)]
    gwnet = GWNet(N, gw_sup).to(dev)
    gwnet.load_state_dict(torch.load(args.gwnet_ckpt, map_location=dev))
    gwnet.eval()

x = torch.from_numpy(test[now - 11:now + 1][None]).to(dev)                      # (1, 12, N, 3)
f = torch.from_numpy(np.array(feats[now:now + 1], dtype=np.float32)).to(dev)    # (1, N, 44)


def timed(fn, reps=20):
    """Chạy fn nhiều lần, trả về (kết quả, thời gian trung bình ms)."""
    with torch.no_grad():
        out = fn()                                 # chạy nháp một lần cho GPU "khởi động"
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        for _ in range(reps):
            out = fn()
        if dev.type == 'cuda':
            torch.cuda.synchronize()
    return out, (time.perf_counter() - t1) / reps * 1000


yb, ms_b = timed(lambda: bigst(x.transpose(1, 2), f)[0])                        # (1, N, 12)
pred_b = (yb[0] * std + mean).cpu().numpy()                                      # (N, 12)
if cai_tien is not None:
    yc, ms_c = timed(lambda: cai_tien(x.transpose(1, 2), f)[0])
    pred_c = (yc[0] * std + mean).cpu().numpy()
if gwnet is not None:
    yg, ms_g = timed(lambda: gwnet(x[..., :2].permute(0, 3, 2, 1))[..., 0].transpose(1, 2))
    pred_g = (yg[0] * std + mean).cpu().numpy()

# ============================== 3. Kết quả ==============================
truth = raw[now + 1:now + 1 + H, node]
persist = np.full(H, raw[now, node])
line('-')
print(f'Tốc độ 1 giờ vừa qua tại cảm biến {args.sensor} (dặm/giờ):')
past = raw[now - 11:now + 1, node]
print('  ' + '  '.join(f'{v:5.1f}' for v in past))
line('-')
hdr = f'{"Sau":>7} {"Giờ":>6} {"Thực tế":>9} {"BigST":>8}' + (f' {"Cải tiến":>9}' if cai_tien is not None else '') + (f' {"GWNET":>8}' if gwnet is not None else '') + f' {"Giữ nguyên":>11}'
print(hdr)
for h in range(H):
    row = f'{(h + 1) * 5:>4} ph {clock(now + 1 + h):>6} {truth[h]:9.1f} {pred_b[node, h]:8.1f}'
    if cai_tien is not None:
        row += f' {pred_c[node, h]:9.1f}'
    if gwnet is not None:
        row += f' {pred_g[node, h]:8.1f}'
    print(row + f' {persist[h]:11.1f}')
line('-')
mae = lambda p: np.abs(p - truth).mean()
print(f'MAE 60 phút tại cảm biến này : BigST {mae(pred_b[node]):.2f}'
      + (f' | Cải tiến {mae(pred_c[node]):.2f}' if cai_tien is not None else '')
      + (f' | GWNET {mae(pred_g[node]):.2f}' if gwnet is not None else '')
      + f' | Giữ nguyên {mae(persist):.2f}')
all_truth = raw[now + 1:now + 1 + H].T                                           # (N, 12)
print(f'MAE trên cả {N} cảm biến     : BigST {np.abs(pred_b - all_truth).mean():.2f}'
      + (f' | Cải tiến {np.abs(pred_c - all_truth).mean():.2f}' if cai_tien is not None else '')
      + (f' | GWNET {np.abs(pred_g - all_truth).mean():.2f}' if gwnet is not None else ''))
print(f'Thời gian dự báo cả {N} cảm biến: BigST {ms_b:.1f} ms'
      + (f' | Cải tiến {ms_c:.1f} ms' if cai_tien is not None else '')
      + (f' | GWNET {ms_g:.1f} ms' if gwnet is not None else '') + ' (trung bình 20 lần)')

# ============================== 4. Hình ==============================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

hist = raw[now - 23:now + 1, node]                                               # 2 giờ quá khứ
tx_h = np.arange(-23, 1) * 5
tx_f = np.arange(1, H + 1) * 5
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.plot(tx_h, hist, color='black', lw=1.8, label='Đã quan sát')
ax.plot(np.r_[0, tx_f], np.r_[hist[-1], truth], color='black', lw=1.8, ls='--', label='Thực tế (chưa biết)')
ax.plot(np.r_[0, tx_f], np.r_[hist[-1], pred_b[node]], color='tab:blue', lw=2.2, marker='o', ms=4,
        label=f'BigST (MAE {mae(pred_b[node]):.2f})')
if cai_tien is not None:
    ax.plot(np.r_[0, tx_f], np.r_[hist[-1], pred_c[node]], color='tab:green', lw=2.2, marker='^', ms=4,
            label=f'BigST cải tiến (MAE {mae(pred_c[node]):.2f})')
if gwnet is not None:
    ax.plot(np.r_[0, tx_f], np.r_[hist[-1], pred_g[node]], color='tab:orange', lw=1.6, marker='s', ms=3,
            label=f'GWNET (MAE {mae(pred_g[node]):.2f})')
ax.axvline(0, color='tab:red', lw=1)
ax.text(1, ax.get_ylim()[1], ' bây giờ ' + clock(now), color='tab:red', va='top')
ax.axvspan(0, tx_f[-1], color='tab:blue', alpha=0.06)
ax.set_xlabel('Phút so với "bây giờ"')
ax.set_ylabel('Tốc độ (dặm/giờ)')
ax.set_title(f'Dự báo 60 phút tới tại cảm biến {args.sensor}')
ax.grid(alpha=0.3)
ax.legend(loc='lower left', fontsize=9)
plt.tight_layout()
out = f'results/demo_{args.sensor}_ngay{day}_{clock(now).replace(":", "h")}.png'
try:
    plt.savefig(out, dpi=200)
except OSError:
    out = out.replace('.png', f'_{int(time.time())}.png')
    plt.savefig(out, dpi=200)
line()
print(f'Đã lưu hình: {out}')
if not args.no_open and sys.platform.startswith('win'):
    os.startfile(os.path.abspath(out))