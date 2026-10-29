
import argparse
import time

import numpy as np
import torch

from common import L_OUT, TrafficData, eval_horizons, save_results

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--data_path', type=str, default='data/pems_sub1000/short_term')
parser.add_argument('--min_t', type=int, default=2015)
parser.add_argument('--lags', type=int, default=3, help='bậc trễ p của VAR')
parser.add_argument('--lambdas', type=float, nargs='+', default=[10, 100, 1e3, 1e4, 1e5, 1e6])
parser.add_argument('--name', type=str, default='VAR')
args = parser.parse_args()
print(args)

dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
data = TrafficData(args.data_path, args.min_t)
N, p = data.N, args.lags


def design(split):
    """X (S, N*p + 1); Y (S, 12*N) = mức thay đổi so với giá trị hiện tại; last (S, 12*N) = giá trị
    hiện tại lặp lại 12 lần để cộng trả lại. Tất cả ở dạng chuẩn hóa."""
    s = torch.from_numpy(data.norm[split][..., 0]).to(dev).double()     # (T, N)
    idx = torch.from_numpy(data.indices(split)).to(dev)
    X = torch.cat([s[idx - k] for k in range(p)] + [torch.ones(len(idx), 1, device=dev, dtype=s.dtype)], 1)
    last = s[idx].repeat(1, L_OUT)
    Y = torch.cat([s[idx + h] for h in range(1, L_OUT + 1)], 1) - last
    return X, Y, last


t0 = time.time()
Xtr, Ytr, _ = design('train')
XtX, XtY = Xtr.T @ Xtr, Xtr.T @ Ytr
del Xtr, Ytr
Xva, _, last_va = design('val')
yva = torch.from_numpy(data.raw['val'][data.indices('val')[:, None] + data.y_off][..., 0]).to(dev).double()

best = None
eye = torch.eye(XtX.shape[0], device=dev, dtype=XtX.dtype)
eye[-1, -1] = 0                                              # không phạt hệ số chặn
for lam in args.lambdas:
    W = torch.linalg.solve(XtX + lam * eye, XtY)            # (N*p+1, 12*N)
    pred = data.denorm(Xva @ W + last_va).reshape(-1, L_OUT, N)
    mae = (pred - yva).abs().mean().item()
    print(f'lambda = {lam:g}: Valid MAE {mae:.4f}')
    if best is None or mae < best[0]:
        best = (mae, lam, W)
mae_val, lam, W = best
fit_time = time.time() - t0
print(f'Chọn lambda = {lam:g} (Valid MAE {mae_val:.4f}); thời gian khớp mô hình {fit_time:.1f} giây')

# ---- kiểm tra: trung bình theo batch 16 giống các mô hình khác ----
Xte, _, last_te = design('test')
pred = data.denorm(Xte @ W + last_te).reshape(-1, L_OUT, N).float()
real = torch.from_numpy(data.raw['test'][data.indices('test')[:, None] + data.y_off][..., 0]).to(dev)
sums, nb = np.zeros((3, L_OUT)), 0
for i in range(0, pred.shape[0], 16):
    sums += eval_horizons(pred[i:i + 16], real[i:i + 16])
    nb += 1
save_results(args.name, sums / nb, {'lags': p, 'lambda': lam, 'num_params': W.numel(),
                                    'fit_sec': f'{fit_time:.1f}'})