"""
Huấn luyện Giai đoạn 2 của BigST (LGSCN) dùng đặc trưng dài hạn đã tính sẵn bởi
precompute_features.py, thay vì tính lại LSFE ở mỗi vòng lặp như run.py.

Mô hình, hàm mất mát và cách đánh giá giữ nguyên như run.py (dùng lại Trainer, BigST).
Khác biệt:
  - đặc trưng dài hạn đọc từ file thay vì tính lại;
  - --min_t: chỉ số đầu tiên của cửa sổ. Mặc định 2015 để cả hai cấu hình (có/không dùng
    đặc trưng dài hạn) được huấn luyện và kiểm tra trên CÙNG một tập mẫu -> so sánh công bằng;
  - đo thời gian mỗi epoch đúng cách và lưu kết quả từng horizon ra file CSV.

Ví dụ:
  # BigST đầy đủ (cấu hình B)
  python run_fast.py --use_long --name B_long --epochs 50
  # Không dùng đặc trưng dài hạn, cùng tập mẫu (cấu hình A)
  python run_fast.py --name A_noLong --epochs 50
  # Ablation: chỉ LSFE (woPFS) / chỉ đặc trưng chu kỳ (woCLT)
  python run_fast.py --use_long --feat_mode long_only --name B_woPFS --epochs 50
  python run_fast.py --use_long --feat_mode period_only --name B_woCLT --epochs 50
"""
import argparse
import copy
import csv
import os
import time

import numpy as np
import torch

import util
from trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--data_path', type=str, default='data/pems_sub1000/short_term')
parser.add_argument('--adjdata', type=str, default='data/pems_sub1000/adj_pems.npy')
parser.add_argument('--feat_path', type=str, default='data/pems_sub1000/features')
parser.add_argument('--name', type=str, default='run', help='tên thí nghiệm (dùng đặt tên file kết quả)')
parser.add_argument('--min_t', type=int, default=2015, help='chỉ số cuối cửa sổ nhỏ nhất')
parser.add_argument('--input_length', type=int, default=12)
parser.add_argument('--output_length', type=int, default=12)
parser.add_argument('--hid_dim', type=int, default=32)
parser.add_argument('--in_dim', type=int, default=3)
parser.add_argument('--num_nodes', type=int, default=1000)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--tau', type=float, default=0.25, help='hệ số nhiệt độ (run.py gốc khai báo nhầm kiểu int)')
parser.add_argument('--random_feature_dim', type=int, default=64)
parser.add_argument('--node_dim', type=int, default=32)
parser.add_argument('--time_dim', type=int, default=32)
parser.add_argument('--time_num', type=int, default=288)
parser.add_argument('--week_num', type=int, default=7)
parser.add_argument('--use_residual', action='store_true', default=True)
parser.add_argument('--use_bn', action='store_true', default=True)
parser.add_argument('--use_spatial', action='store_true', help='thêm hàm mất mát không gian (Eq. 20)')
parser.add_argument('--use_long', action='store_true', help='dùng đặc trưng dài hạn đã tính sẵn')
parser.add_argument('--feat_mode', type=str, default='all', choices=['all', 'long_only', 'period_only'],
                    help='ablation: all = h_long + h_pe; long_only = bỏ đặc trưng chu kỳ (woPFS); '
                         'period_only = bỏ đặc trưng LSFE (woCLT). Phần bị bỏ được đặt bằng 0.')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--grad_clip', type=float, default=5)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--milestones', type=int, nargs='+', default=[80, 100])
parser.add_argument('--patience', type=int, default=30)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
args.device = torch.device(args.device)
os.makedirs('results', exist_ok=True)
os.makedirs('checkpoint', exist_ok=True)

# ---------- dữ liệu ----------
data = {s: np.load(os.path.join(args.data_path, f'{s}.npy')).astype(np.float32) for s in ['train', 'val', 'test']}
scaler = util.StandardScaler(mean=data['train'][..., 0].mean(), std=data['train'][..., 0].std())
for s in data:
    data[s][..., 0] = scaler.transform(data[s][..., 0])
    print('*' * 10, s, data[s].shape, '*' * 10)

feats = None
if args.use_long:
    feats = {s: np.load(os.path.join(args.feat_path, f'{s}.npy'), mmap_mode='r') for s in data}

x_off = np.arange(-(args.input_length - 1), 1)       # 12 bước gần nhất
y_off = np.arange(1, args.output_length + 1)         # 12 bước tiếp theo


def batches(split, shuffle):
    """Sinh (x, y, feat) theo batch. idx = chỉ số cuối cửa sổ lịch sử."""
    T = data[split].shape[0]
    idx = np.arange(max(args.min_t, args.input_length - 1), T - args.output_length)
    if shuffle:
        idx = np.random.permutation(idx)
    for i in range(0, len(idx), args.batch_size):
        b = idx[i:i + args.batch_size]
        x = torch.from_numpy(data[split][b[:, None] + x_off]).to(args.device).transpose(1, 2)  # (B, N, T, F)
        y = torch.from_numpy(data[split][b[:, None] + y_off]).to(args.device).transpose(1, 2)
        f = None
        if feats is not None:
            f = torch.from_numpy(np.array(feats[split][b], dtype=np.float32)).to(args.device)  # (B, N, 44)
            if args.feat_mode == 'long_only':        # woPFS: bỏ 12 chiều đặc trưng chu kỳ
                f[..., -args.output_length:] = 0
            elif args.feat_mode == 'period_only':    # woCLT: bỏ 32 chiều đặc trưng LSFE
                f[..., :-args.output_length] = 0
        yield x, y[:, :, :, 0], f


# ---------- mô hình ----------
adj_mx = util.load_adj(args.adjdata)
supports = [torch.tensor(i).to(args.device) for i in adj_mx]
edge_indices = torch.nonzero(supports[0] > 0)
trainer = Trainer(args, scaler, supports, edge_indices)
n_params = sum(p.numel() for p in trainer.model.parameters())
print(f'Số tham số: {n_params}')

# ---------- huấn luyện ----------
best_val, best_state, best_epoch, wait = np.inf, None, 0, 0
epoch_times = []
if args.device.type == 'cuda':
    torch.cuda.reset_peak_memory_stats(args.device)
for ep in range(1, args.epochs + 1):
    t1 = time.time()
    tr = [trainer.train(x, y, f) for x, y, f in batches('train', True)]
    trainer.scheduler.step()
    t_train = time.time() - t1
    epoch_times.append(t_train)

    va = [trainer.eval(x, y, f) for x, y, f in batches('val', False)]
    tr, va = np.mean(tr, axis=0), np.mean(va, axis=0)
    print(f'Epoch {ep:03d} | train {t_train:.1f}s | Train MAE {tr[0]:.4f} | '
          f'Valid MAE {va[0]:.4f}, MAPE {va[1]:.4f}, RMSE {va[2]:.4f}', flush=True)

    if va[0] < best_val:
        best_val, best_epoch, wait = va[0], ep, 0
        best_state = copy.deepcopy(trainer.model.state_dict())
    else:
        wait += 1
        if wait >= args.patience:
            print('Dừng sớm (early stopping)')
            break

# ---------- kiểm tra ----------
trainer.model.load_state_dict(best_state)
torch.save(best_state, f'checkpoint/bigst_{args.name}.pth')
H = args.output_length
sums = np.zeros((3, H))
n_batches = 0
t1 = time.time()
for x, y, f in batches('test', False):
    m = trainer.eval(x, y, f, flag='horizon')   # (3 chỉ số, 12 horizon)
    sums += np.array(m)
    n_batches += 1
test_time = time.time() - t1
res = sums / n_batches   # trung bình theo batch, giống run.py

print(f'\nEpoch tốt nhất: {best_epoch}, Valid MAE {best_val:.4f}')
for h in range(H):
    print(f'Horizon {h + 1:2d}: Test MAE {res[0, h]:.4f}, MAPE {res[1, h]:.4f}, RMSE {res[2, h]:.4f}')
print(f'Trung bình 12 horizon: Test MAE {res[0].mean():.4f}, MAPE {res[1].mean():.4f}, RMSE {res[2].mean():.4f}')
print(f'Thời gian huấn luyện trung bình: {np.mean(epoch_times):.1f} giây/epoch; thời gian test: {test_time:.1f} giây')

with open(f'results/{args.name}.csv', 'w', newline='', encoding='utf-8') as fcsv:
    w = csv.writer(fcsv)
    w.writerow(['horizon', 'MAE', 'MAPE', 'RMSE'])
    for h in range(H):
        w.writerow([h + 1, f'{res[0, h]:.4f}', f'{res[1, h]:.4f}', f'{res[2, h]:.4f}'])
    w.writerow(['avg', f'{res[0].mean():.4f}', f'{res[1].mean():.4f}', f'{res[2].mean():.4f}'])
    w.writerow([])
    w.writerow(['best_epoch', best_epoch])
    w.writerow(['train_sec_per_epoch', f'{np.mean(epoch_times):.1f}'])
    w.writerow(['num_params', n_params])
    if args.device.type == 'cuda':
        w.writerow(['peak_gpu_MB', f'{torch.cuda.max_memory_allocated(args.device) / 2 ** 20:.1f}'])
    w.writerow(['test_sec', f'{test_time:.1f}'])
print(f'Đã lưu kết quả: results/{args.name}.csv')