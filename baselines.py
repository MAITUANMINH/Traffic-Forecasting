"""
Các mô hình cơ sở (baseline) đơn giản, đánh giá trên CÙNG tập mẫu test với run_fast.py
(cửa sổ kết thúc tại t >= min_t = 2015, dự báo 12 bước t+1..t+12), cùng cách tính
MAE/MAPE/RMSE có mặt nạ (bỏ qua giá trị 0) như metrics.py, lấy trung bình theo batch 16.

  1. HA (Historical Average): với mỗi nút và mỗi khung 5 phút trong ngày (288 khung),
     dự báo = trung bình tốc độ của khung đó trên toàn bộ tập train.
  2. HA-7 ngày: dự báo = trung bình tốc độ cùng khung giờ trong 7 ngày ngay trước đó.
  3. Persistence (giữ nguyên): dự báo 12 bước tới = giá trị quan sát cuối cùng.

Ví dụ:
    python baselines.py --data_path data/pems_sub1000/short_term
"""
import argparse
import csv
import os

import warnings

import numpy as np

warnings.filterwarnings('ignore', category=RuntimeWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='data/pems_sub1000/short_term')
parser.add_argument('--min_t', type=int, default=2015)
parser.add_argument('--output_length', type=int, default=12)
parser.add_argument('--time_num', type=int, default=288)
parser.add_argument('--batch_size', type=int, default=16)
args = parser.parse_args()
os.makedirs('results', exist_ok=True)

H, P = args.output_length, args.time_num
train = np.load(os.path.join(args.data_path, 'train.npy')).astype(np.float64)
test = np.load(os.path.join(args.data_path, 'test.npy')).astype(np.float64)
T, N, _ = test.shape


def slot_of(tod):
    """Chỉ số khung giờ 0..287 từ kênh 'thời điểm trong ngày' (giá trị 0..1)."""
    return np.floor(tod * P + 1e-6).astype(int) % P


# ---- các chỉ số có mặt nạ, giống metrics.py (null_val = 0) ----
def masked(pred, real, kind):
    if kind == 'mape':
        real = np.where(real < 0.01, 0.0, real)
    mask = (real != 0).astype(np.float64)
    if mask.mean() == 0:
        return 0.0
    mask /= mask.mean()
    if kind == 'mae':
        loss = np.abs(pred - real)
    elif kind == 'mape':
        with np.errstate(divide='ignore', invalid='ignore'):
            loss = np.abs(pred - real) / real
    else:
        loss = (pred - real) ** 2
    loss = np.nan_to_num(loss * mask, nan=0.0, posinf=0.0, neginf=0.0)
    return np.sqrt(loss.mean()) if kind == 'rmse' else loss.mean()


# ---- HA: bảng trung bình theo (khung giờ, nút) từ tập train ----
train_slot = slot_of(train[:, 0, 1])                       # khung giờ của từng bước train
ha_table = np.zeros((P, N))
for s in range(P):
    v = train[train_slot == s, :, 0]                       # (số ngày, N)
    v = np.where(v == 0, np.nan, v)                        # bỏ giá trị 0 (thiếu dữ liệu)
    ha_table[s] = np.nan_to_num(np.nanmean(v, axis=0), nan=np.nanmean(train[..., 0]))

idx = np.arange(max(args.min_t, 0), T - H)
y_off = np.arange(1, H + 1)
day_offsets = np.arange(1, 8) * P                          # 1..7 ngày trước

methods = {
    'HA': lambda b: ha_table[slot_of(test[b[:, None] + y_off, 0, 1])],                # (B, H, N)
    'HA_7ngay': lambda b: np.nan_to_num(np.nanmean(                                   # bỏ giá trị 0
        [np.where(test[b[:, None] + y_off - d, :, 0] == 0, np.nan, test[b[:, None] + y_off - d, :, 0])
         for d in day_offsets], axis=0)),                                              # (B, H, N)
    'Persistence': lambda b: np.repeat(test[b, None, :, 0], H, axis=1),               # (B, H, N)
}

for name, predict in methods.items():
    sums, nb = np.zeros((3, H)), 0
    for i in range(0, len(idx), args.batch_size):
        b = idx[i:i + args.batch_size]
        pred = predict(b)
        real = test[b[:, None] + y_off, :, 0]
        for h in range(H):
            sums[:, h] += [masked(pred[:, h], real[:, h], k) for k in ('mae', 'mape', 'rmse')]
        nb += 1
    res = sums / nb
    print(f'\n== {name} ==')
    for h in (2, 5, 11):
        print(f'Horizon {h + 1:2d}: MAE {res[0, h]:.4f}, MAPE {res[1, h]:.4f}, RMSE {res[2, h]:.4f}')
    print(f'Trung bình: MAE {res[0].mean():.4f}, MAPE {res[1].mean():.4f}, RMSE {res[2].mean():.4f}')
    with open(f'results/{name}.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['horizon', 'MAE', 'MAPE', 'RMSE'])
        for h in range(H):
            w.writerow([h + 1, f'{res[0, h]:.4f}', f'{res[1, h]:.4f}', f'{res[2, h]:.4f}'])
        w.writerow(['avg', f'{res[0].mean():.4f}', f'{res[1].mean():.4f}', f'{res[2].mean():.4f}'])
print('\nĐã lưu kết quả vào thư mục results/')