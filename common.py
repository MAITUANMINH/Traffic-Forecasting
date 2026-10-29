
import copy
import csv
import os
import time

import numpy as np
import torch

L_IN, L_OUT = 12, 12          # 12 bước quá khứ -> 12 bước tương lai (5 phút/bước)


# ============================ Dữ liệu ============================
class TrafficData:
    """Đọc train/val/test (T, N, 3): kênh 0 tốc độ, 1 thời điểm trong ngày, 2 ngày trong tuần.

    x trả về đã chuẩn hóa kênh tốc độ (trung bình, độ lệch chuẩn của tập train);
    y trả về là tốc độ GỐC (dặm/giờ) để tính sai số đúng đơn vị.
    """

    def __init__(self, data_path, min_t=2015):
        self.raw = {s: np.load(os.path.join(data_path, f'{s}.npy')).astype(np.float32)
                    for s in ['train', 'val', 'test']}
        self.mean = float(self.raw['train'][..., 0].mean())
        self.std = float(self.raw['train'][..., 0].std())
        self.norm = {}
        for s, a in self.raw.items():
            b = a.copy()
            b[..., 0] = (b[..., 0] - self.mean) / self.std
            self.norm[s] = b
        self.min_t = min_t
        self.N = self.raw['train'].shape[1]
        self.x_off = np.arange(-(L_IN - 1), 1)
        self.y_off = np.arange(1, L_OUT + 1)

    def indices(self, split):
        T = self.raw[split].shape[0]
        return np.arange(max(self.min_t, L_IN - 1), T - L_OUT)

    def batches(self, split, bs, shuffle, in_dim, device, limit=0):
        """Sinh (x, y): x (B, 12, N, in_dim) đã chuẩn hóa; y (B, 12, N) tốc độ gốc.
        limit > 0: chỉ lấy `limit` batch đầu (chế độ chạy thử --quick)."""
        idx = self.indices(split)
        if shuffle:
            idx = np.random.permutation(idx)
        if limit:
            idx = idx[:limit * bs]
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            x = self.norm[split][b[:, None] + self.x_off][..., :in_dim]
            y = self.raw[split][b[:, None] + self.y_off][..., 0]
            yield torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)

    def denorm(self, z):
        return z * self.std + self.mean


def load_adj(path):
    return np.load(path).astype(np.float32)


def random_walk(adj):
    """Ma trận chuyển tiếp D^-1 A (giống util.asym_adj của code gốc)."""
    d = adj.sum(1)
    d_inv = np.where(d > 0, 1.0 / d, 0.0)
    return (d_inv[:, None] * adj).astype(np.float32)


# ============================ Hàm đo (giống metrics.py) ============================
def _mask(labels):
    mask = (labels != 0).float()
    mask = mask / torch.mean(mask)
    return torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)


def masked_mae(pred, real):
    loss = torch.abs(pred - real) * _mask(real)
    return torch.mean(torch.nan_to_num(loss))


def masked_mape(pred, real):
    real = torch.where(real < 1e-2, torch.zeros_like(real), real)
    loss = torch.abs(pred - real) / real * _mask(real)
    return torch.mean(torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0))


def masked_rmse(pred, real):
    loss = (pred - real) ** 2 * _mask(real)
    return torch.sqrt(torch.mean(torch.nan_to_num(loss)))


def eval_horizons(pred, real):
    """pred, real: (B, 12, N) -> mảng (3, 12): MAE, MAPE, RMSE theo từng horizon."""
    out = np.zeros((3, L_OUT))
    for h in range(L_OUT):
        p, r = pred[:, h], real[:, h]
        out[:, h] = [masked_mae(p, r).item(), masked_mape(p, r).item(), masked_rmse(p, r).item()]
    return out


# ============================ Ghi kết quả ============================
def save_results(name, res, extra):
    """res: (3, 12). Cùng định dạng CSV với run_fast.py để summarize_results.py đọc được."""
    os.makedirs('results', exist_ok=True)
    print()
    for h in range(L_OUT):
        print(f'Horizon {h + 1:2d}: Test MAE {res[0, h]:.4f}, MAPE {res[1, h]:.4f}, RMSE {res[2, h]:.4f}')
    print(f'Trung bình 12 horizon: Test MAE {res[0].mean():.4f}, MAPE {res[1].mean():.4f}, RMSE {res[2].mean():.4f}')
    with open(f'results/{name}.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['horizon', 'MAE', 'MAPE', 'RMSE'])
        for h in range(L_OUT):
            w.writerow([h + 1, f'{res[0, h]:.4f}', f'{res[1, h]:.4f}', f'{res[2, h]:.4f}'])
        w.writerow(['avg', f'{res[0].mean():.4f}', f'{res[1].mean():.4f}', f'{res[2].mean():.4f}'])
        w.writerow([])
        for k, v in extra.items():
            w.writerow([k, v])
    print(f'Đã lưu kết quả: results/{name}.csv')


# ============================ Huấn luyện chung ============================
def train_and_test(model, forward, data, args, device, in_dim, lr_milestones=None):
    """Vòng huấn luyện dùng chung: Adam, mất mát MAE có mặt nạ trên đơn vị gốc, cắt gradient 5,
    dừng sớm theo Valid MAE, rồi kiểm tra mô hình tốt nhất trên tập test.

    forward(model, x, y_norm, step, train) -> dự báo ĐÃ CHUẨN HÓA, shape (B, 12, N).
    y_norm (nhãn đã chuẩn hóa) chỉ dùng cho mô hình có teacher forcing (DCRNN).
    """
    os.makedirs('checkpoint', exist_ok=True)
    lim = 20 if getattr(args, 'quick', False) else 0          # --quick: 20 batch mỗi tập để thử nhanh
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    sched = (torch.optim.lr_scheduler.MultiStepLR(opt, milestones=lr_milestones, gamma=0.1)
             if lr_milestones else None)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Số tham số: {n_params}')
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    best_val, best_state, best_epoch, wait, step = np.inf, None, 0, 0, 0
    epoch_times = []
    start_ep = 1
    ck_path = f'checkpoint/{args.name}_dangchay.pth'      # lưu sau MỖI epoch để chạy tiếp được
    if getattr(args, 'resume', False) and os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        if sched and ck['sched'] is not None:
            sched.load_state_dict(ck['sched'])
        best_val, best_state, best_epoch = ck['best_val'], ck['best_state'], ck['best_epoch']
        wait, step, epoch_times = ck['wait'], ck['step'], ck['epoch_times']
        np.random.set_state(ck['np_rng'])
        start_ep = ck['epoch'] + 1
        print(f'Chạy tiếp từ epoch {start_ep} (epoch tốt nhất đến giờ: {best_epoch}, Valid MAE {best_val:.4f})')
    elif getattr(args, 'resume', False):
        print(f'Không thấy {ck_path}, chạy lại từ đầu')

    for ep in range(start_ep, args.epochs + 1):
        if wait >= args.patience:
            break
        model.train()
        t1 = time.time()
        tr = []
        for x, y in data.batches('train', args.batch_size, True, in_dim, device, lim):
            opt.zero_grad()
            y_norm = (y - data.mean) / data.std
            pred = data.denorm(forward(model, x, y_norm, step, True))
            loss = masked_mae(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            tr.append(loss.item())
            step += 1
        if sched:
            sched.step()
        t_train = time.time() - t1
        epoch_times.append(t_train)

        model.eval()
        va = []
        with torch.no_grad():
            for x, y in data.batches('val', 16, False, in_dim, device, lim):
                pred = data.denorm(forward(model, x, None, step, False))
                va.append([masked_mae(pred, y).item(), masked_mape(pred, y).item(), masked_rmse(pred, y).item()])
        va = np.mean(va, axis=0)
        print(f'Epoch {ep:03d} | train {t_train:.1f}s | Train MAE {np.mean(tr):.4f} | '
              f'Valid MAE {va[0]:.4f}, MAPE {va[1]:.4f}, RMSE {va[2]:.4f}', flush=True)
        if va[0] < best_val:
            best_val, best_epoch, wait = va[0], ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
        torch.save({'epoch': ep, 'model': model.state_dict(), 'opt': opt.state_dict(),
                    'sched': sched.state_dict() if sched else None,
                    'best_val': best_val, 'best_state': best_state, 'best_epoch': best_epoch,
                    'wait': wait, 'step': step, 'epoch_times': epoch_times,
                    'np_rng': np.random.get_state()}, ck_path)
        if wait >= args.patience:
            print('Dừng sớm (early stopping)')
            break

    model.load_state_dict(best_state)
    torch.save(best_state, f'checkpoint/{args.name}.pth')
    model.eval()
    sums, nb = np.zeros((3, L_OUT)), 0
    t1 = time.time()
    with torch.no_grad():
        for x, y in data.batches('test', 16, False, in_dim, device, lim):     # batch 16 giống run.py
            pred = data.denorm(forward(model, x, None, step, False))
            sums += eval_horizons(pred, y)
            nb += 1
    test_time = time.time() - t1
    res = sums / nb
    print(f'\nEpoch tốt nhất: {best_epoch}, Valid MAE {best_val:.4f}')
    peak = torch.cuda.max_memory_allocated(device) / 2 ** 20 if device.type == 'cuda' else float('nan')
    print(f'Thời gian huấn luyện TB: {np.mean(epoch_times):.1f} giây/epoch; test {test_time:.1f} giây; '
          f'bộ nhớ GPU đỉnh {peak:.0f} MB')
    save_results(args.name, res, {'best_epoch': best_epoch,
                                  'train_sec_per_epoch': f'{np.mean(epoch_times):.1f}',
                                  'num_params': n_params,
                                  'peak_gpu_MB': f'{peak:.1f}',
                                  'test_sec': f'{test_time:.1f}'})
    return res