"""
Thống kê dữ liệu (tương ứng Bảng 1 của bài báo) cho bộ California đầy đủ và tập con.
Đọc theo từng khối (memmap) nên không cần nhiều RAM.

Ví dụ:
    python dataset_stats.py
"""
import csv
import os

import numpy as np

DATASETS = {
    'California (đầy đủ)': ('data/pems_data/pems_data/short_term', 'data/pems_data/pems_data/adj_pems.npy'),
    'California (tập con 1.000 nút)': ('data/pems_sub1000/short_term', 'data/pems_sub1000/adj_pems.npy'),
}
STEPS_PER_DAY = 288  # 5 phút một bước
os.makedirs('results', exist_ok=True)


def stats(data_dir, adj_path):
    splits = {s: np.load(os.path.join(data_dir, f'{s}.npy'), mmap_mode='r') for s in ['train', 'val', 'test']}
    N = splits['train'].shape[1]
    T = {s: a.shape[0] for s, a in splits.items()}
    total_T = sum(T.values())

    # tốc độ: tính theo khối để tiết kiệm RAM; giá trị 0 coi là thiếu dữ liệu
    n_all = n_zero = 0
    s1 = s2 = 0.0
    vmin, vmax = np.inf, -np.inf
    for a in splits.values():
        for i in range(0, a.shape[0], 1000):
            v = np.asarray(a[i:i + 1000, :, 0], dtype=np.float64)
            nz = v[v != 0]
            n_all += v.size
            n_zero += v.size - nz.size
            s1 += nz.sum(); s2 += (nz ** 2).sum()
            if nz.size:
                vmin, vmax = min(vmin, nz.min()), max(vmax, nz.max())
    n_nz = n_all - n_zero
    mean = s1 / n_nz
    std = np.sqrt(s2 / n_nz - mean ** 2)

    # đồ thị: số cạnh khác 0 (không tính đường chéo), bậc trung bình
    adj = np.load(adj_path, mmap_mode='r')
    n_edges = 0
    for i in range(0, N, 500):
        block = np.asarray(adj[i:i + 500]) != 0
        rows = np.arange(block.shape[0])
        n_edges += int(block.sum() - block[rows, i + rows].sum())   # bỏ đường chéo (tự nối)
    return {
        'Số nút': N,
        'Số bước thời gian': total_T,
        'Train / Val / Test (bước)': f"{T['train']} / {T['val']} / {T['test']}",
        'Số bản ghi tốc độ': N * total_T,
        'Khoảng thời gian (ngày)': f'{total_T / STEPS_PER_DAY:.1f}',
        'Tỉ lệ giá trị 0 (thiếu)': f'{n_zero / n_all:.4%}',
        'Tốc độ TB ± độ lệch chuẩn': f'{mean:.2f} ± {std:.2f}',
        'Tốc độ nhỏ nhất / lớn nhất': f'{vmin:.1f} / {vmax:.1f}',
        'Số cạnh trong ma trận kề': int(n_edges),
        'Bậc trung bình': f'{n_edges / N:.2f}',
    }


results = {}
for name, (d, a) in DATASETS.items():
    if not os.path.exists(d):
        print(f'Bỏ qua {name}: không thấy {d}')
        continue
    print(f'Đang thống kê {name} ...', flush=True)
    results[name] = stats(d, a)

keys = list(next(iter(results.values())).keys())
with open('results/dataset_stats.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['Chỉ số'] + list(results.keys()))
    for k in keys:
        w.writerow([k] + [results[n][k] for n in results])
print()
for k in keys:
    print(f'{k:30s} | ' + ' | '.join(str(results[n][k]) for n in results))
print('\nĐã lưu: results/dataset_stats.csv')