
import argparse
import csv
import os

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='data/pems_sub1000/short_term')
parser.add_argument('--adjdata', type=str, default='data/pems_sub1000/adj_pems.npy')
parser.add_argument('--acf_nodes', type=int, default=200, help='số cảm biến dùng để tính tự tương quan')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
os.makedirs('results', exist_ok=True)
rng = np.random.default_rng(args.seed)
P = 288                                                     # số bước 5 phút trong một ngày

train = np.load(os.path.join(args.data_path, 'train.npy')).astype(np.float32)   # (T, N, 3)
T, N, _ = train.shape
speed = train[..., 0]                                       # (T, N)
slot = np.floor(train[:, 0, 1] * P + 1e-6).astype(int) % P  # khung 5 phút trong ngày (0..287)
dow_raw = train[:, 0, 2]
dow = np.round(dow_raw * 7).astype(int) % 7 if dow_raw.max() <= 1.0 else dow_raw.astype(int) % 7
stats = []

v = speed.ravel()
q = np.percentile(v, [1, 5, 25, 50, 75, 95, 99])
stats += [('Số bước train', T), ('Số cảm biến', N),
          ('Tốc độ trung bình (dặm/giờ)', f'{v.mean():.2f}'), ('Độ lệch chuẩn', f'{v.std():.2f}'),
          ('Trung vị', f'{q[3]:.1f}'), ('Phân vị 5% / 95%', f'{q[1]:.1f} / {q[5]:.1f}'),
          ('Tỉ lệ bản ghi dưới 40 dặm/giờ (ùn tắc)', f'{(v < 40).mean():.2%}'),
          ('Tỉ lệ giá trị 0 (thiếu)', f'{(v == 0).mean():.2%}')]
plt.figure(figsize=(6, 4))
plt.hist(v, bins=100, color='#2E86AB')
plt.axvline(v.mean(), color='k', ls='--', lw=1, label=f'trung bình {v.mean():.1f}')
plt.xlabel('Tốc độ (dặm/giờ)'); plt.ylabel('Số bản ghi'); plt.yscale('log')
plt.title('Phân bố tốc độ trên tập train (trục dọc thang log)')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig('results/eda_phan_bo.png', dpi=200); plt.close()

# ---------- . Tốc độ theo giờ trong ngày, tách theo mã ngày trong tuần ----------
plt.figure(figsize=(7, 4))
hours = np.arange(P) / 12
for d in range(7):
    m = dow == d
    if m.sum() == 0:
        continue
    prof = np.array([speed[m & (slot == s)].mean() if (m & (slot == s)).any() else np.nan for s in range(P)])
    plt.plot(hours, prof, lw=1.2, label=f'mã ngày {d}')
    morning = prof[(hours >= 6) & (hours < 9)]
    stats.append((f'Tốc độ TB 6h-9h, mã ngày {d}', f'{np.nanmean(morning):.2f}'))
allday = np.array([speed[slot == s].mean() for s in range(P)])
stats += [('Giờ có tốc độ TB thấp nhất', f'{hours[np.argmin(allday)]:.1f}h ({allday.min():.1f} dặm/giờ)'),
          ('Giờ có tốc độ TB cao nhất', f'{hours[np.argmax(allday)]:.1f}h ({allday.max():.1f} dặm/giờ)')]
plt.xlabel('Giờ trong ngày'); plt.ylabel('Tốc độ trung bình (dặm/giờ)')
plt.title('Tốc độ trung bình theo giờ, mỗi đường một ngày trong tuần')
plt.xticks(range(0, 25, 3)); plt.grid(alpha=0.3); plt.legend(fontsize=8, ncol=2); plt.tight_layout()
plt.savefig('results/eda_theo_gio.png', dpi=200); plt.close()

# ---------- 3. Tự tương quan theo độ trễ (tới 8 ngày) ----------
nodes = rng.choice(N, size=min(args.acf_nodes, N), replace=False)
max_lag = 8 * P
acfs = []
for n in nodes:
    x = speed[:, n] - speed[:, n].mean()
    f = np.fft.rfft(x, n=2 * T)
    ac = np.fft.irfft(f * np.conj(f))[:max_lag + 1]
    acfs.append(ac / ac[0])
acf = np.mean(acfs, axis=0)
for lag, label in [(1, '5 phút'), (12, '1 giờ'), (P // 2, '12 giờ'), (P, '1 ngày'), (7 * P, '7 ngày')]:
    stats.append((f'Tự tương quan TB ở độ trễ {label}', f'{acf[lag]:.3f}'))
plt.figure(figsize=(7, 4))
plt.plot(np.arange(max_lag + 1) / P, acf, color='#2E86AB', lw=1)
for d, c in [(1, '#E07A2F'), (7, '#C0392B')]:
    plt.axvline(d, color=c, ls='--', lw=1, label=f'{d} ngày ({d * P} bước)')
plt.xlabel('Độ trễ (ngày)'); plt.ylabel('Hệ số tự tương quan')
plt.title(f'Tự tương quan trung bình của {len(nodes)} cảm biến')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig('results/eda_tu_tuong_quan.png', dpi=200); plt.close()

# ----------Tương quan giữa cặp cảm biến kề nhau và cặp ngẫu nhiên ----------
adj = np.load(args.adjdata)
ii, jj = np.nonzero(adj)
keep = ii != jj
ii, jj = ii[keep], jj[keep]
z = (speed - speed.mean(0)) / (speed.std(0) + 1e-6)          # chuẩn hóa từng cảm biến
def corr(a, b):
    return (z[:, a] * z[:, b]).mean(0)
c_adj = corr(ii, jj)
ri, rj = rng.integers(0, N, len(ii)), rng.integers(0, N, len(ii))
ok = ri != rj
c_rand = corr(ri[ok], rj[ok])
stats += [('Số cặp kề nhau', len(ii)),
          ('Tương quan TB, cặp kề nhau', f'{c_adj.mean():.3f}'),
          ('Tương quan TB, cặp ngẫu nhiên', f'{c_rand.mean():.3f}')]
plt.figure(figsize=(6, 4))
bins = np.linspace(-0.2, 1, 61)
plt.hist(c_rand, bins=bins, alpha=0.6, density=True, color='#8A939E', label=f'cặp ngẫu nhiên (TB {c_rand.mean():.2f})')
plt.hist(c_adj, bins=bins, alpha=0.6, density=True, color='#2E86AB', label=f'cặp kề nhau (TB {c_adj.mean():.2f})')
plt.xlabel('Hệ số tương quan Pearson của tốc độ'); plt.ylabel('Mật độ')
plt.title('Tương quan tốc độ giữa hai cảm biến')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig('results/eda_tuong_quan.png', dpi=200); plt.close()

# ---------- In và lưu ----------
with open('results/eda_stats.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['Chỉ số', 'Giá trị'])
    w.writerows(stats)
for k, val in stats:
    print(f'{k:45s} {val}')
print('\nĐã lưu: results/eda_phan_bo.png, eda_theo_gio.png, eda_tu_tuong_quan.png, eda_tuong_quan.png, eda_stats.csv')