
"""
Tính sẵn đặc trưng dài hạn của BigST một lần rồi lưu ra đĩa (mục 3.2.3 của bài báo).

Với mỗi thời điểm t (cuối cửa sổ lịch sử), script tính:
  - h_long (nhid chiều): đầu ra của LSFE (linear_transformer đã huấn luyện ở Giai đoạn 1)
    trên cửa sổ dài L = 2016 bước (1 tuần) kết thúc tại t;
  - h_pe (12 chiều): đặc trưng chu kỳ = trung bình tốc độ (đã chuẩn hóa) của cùng khung giờ
    t+1..t+12 trong 7 ngày trước (giống hàm sample_period trong run.py).
Kết quả: <out>/<split>.npy có dạng (T, N, nhid + 12), dòng t ứng với cửa sổ kết thúc tại t
(các dòng t < L-1 để 0, không dùng).

Code gốc (run.py --use_long) tính lại LSFE cho TỪNG mẫu ở MỖI vòng lặp của MỖI epoch;
tính sẵn một lần giúp Giai đoạn 2 chạy nhanh gần bằng cấu hình không dùng đặc trưng dài hạn.

Ví dụ:
    python precompute_features.py --data_path data/pems_sub1000/short_term --num_nodes 1000
"""
import argparse
import os
import time

import numpy as np
import torch

from preprocess.model import linear_transformer

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--data_path", type=str, default="data/pems_sub1000/short_term")
parser.add_argument("--checkpoint", type=str, default="checkpoint/linear_transformer.pth")
parser.add_argument("--out", type=str, default=None, help="mặc định: <data_path>/../features")
parser.add_argument("--num_nodes", type=int, default=1000)
parser.add_argument("--long_length", type=int, default=2016, help="độ dài cửa sổ dài hạn")
parser.add_argument("--output_length", type=int, default=12)
parser.add_argument("--in_dim", type=int, default=3)
parser.add_argument("--nhid", type=int, default=32)
parser.add_argument("--time_num", type=int, default=288, help="số bước trong một ngày")
args = parser.parse_args()

device = torch.device(args.device)
L = args.long_length
out_dir = args.out or os.path.join(os.path.dirname(os.path.normpath(args.data_path)), "features")
os.makedirs(out_dir, exist_ok=True)

# 1. Đọc dữ liệu, chuẩn hóa kênh tốc độ bằng mean/std của tập train (giống util.load_dataset)
data = {s: np.load(os.path.join(args.data_path, f"{s}.npy")).astype(np.float32)
        for s in ["train", "val", "test"]}
mean, std = data["train"][..., 0].mean(), data["train"][..., 0].std()
for s in data:
    data[s][..., 0] = (data[s][..., 0] - mean) / std
print(f"mean={mean:.4f}, std={std:.4f}")

# 2. Nạp LSFE đã huấn luyện ở Giai đoạn 1
model = linear_transformer(L, args.output_length, args.in_dim, args.num_nodes, args.nhid, 0.3)
model.load_state_dict(torch.load(args.checkpoint, map_location=device))
model.to(device).eval()

# Vị trí các bước dùng cho đặc trưng chu kỳ (giống sample_period trong run.py):
# lấy 12 bước đầu của mỗi ngày trong cửa sổ -> cùng khung giờ với t+1..t+12 ở các ngày trước
period_offsets = [np.arange(i, i + args.output_length) for i in range(0, L, args.time_num)]

for split in ["train", "val", "test"]:
    arr = data[split]  # (T, N, F)
    T, N, _ = arr.shape
    # ghi thẳng ra đĩa (memmap) để không chiếm vài GB RAM
    feats = np.lib.format.open_memmap(os.path.join(out_dir, f"{split}.npy"), mode="w+",
                                      dtype=np.float32, shape=(T, N, args.nhid + args.output_length))
    t0 = time.time()
    with torch.no_grad():
        for t in range(L - 1, T):
            window = arr[t - L + 1:t + 1]  # (L, N, F)
            # LSFE: xử lý từng cửa sổ riêng (B = 1) giống hệt run.py
            x = torch.from_numpy(window).to(device).transpose(0, 1).unsqueeze(0)  # (1, N, L, F)
            _, h_long = model(x)  # (1, N, nhid)
            # đặc trưng chu kỳ: trung bình qua các ngày, kênh tốc độ đã chuẩn hóa
            h_pe = np.mean([window[o, :, 0] for o in period_offsets], axis=0).T  # (N, 12)
            feats[t, :, :args.nhid] = h_long[0].cpu().numpy()
            feats[t, :, args.nhid:] = h_pe
            if (t - L + 1) % 1000 == 0:
                done = t - L + 2
                total = T - L + 1
                eta = (time.time() - t0) / done * (total - done)
                print(f"[{split}] {done}/{total} cửa sổ, còn khoảng {eta / 60:.1f} phút", flush=True)
    shape = feats.shape
    feats.flush()
    del feats
    print(f"[{split}] xong: {shape}, {time.time() - t0:.1f} giây", flush=True)

print(f"Đặc trưng đã lưu tại: {out_dir}")



