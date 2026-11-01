"""
Cắt một tập con N nút liền kề nhau trên mạng đường từ dữ liệu California (PeMS) của BigST.
Chọn nút bằng BFS từ nút có nhiều kết nối nhất -> tập con là một vùng liên thông.
Ví dụ: python make_subset.py --num_nodes 1000
"""
import argparse
import os
from collections import deque

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--src", default="data/pems_data/pems_data", help="thư mục dữ liệu gốc")
parser.add_argument("--num_nodes", type=int, default=1000, help="số nút muốn giữ lại")
parser.add_argument("--out", default=None, help="thư mục đầu ra (mặc định data/pems_sub<N>)")
args = parser.parse_args()

out_dir = args.out or f"data/pems_sub{args.num_nodes}"
os.makedirs(os.path.join(out_dir, "short_term"), exist_ok=True)

# 1. Đọc ma trận kề (mmap: không nạp toàn bộ vào RAM)
adj = np.load(os.path.join(args.src, "adj_pems.npy"), mmap_mode="r")
N_total = adj.shape[0]
print(f"Ma trận kề gốc: {adj.shape}")

# 2. Xây danh sách kề (coi đồ thị là vô hướng)
neighbors = [set() for _ in range(N_total)]
for i in range(N_total):
    for j in np.nonzero(adj[i])[0]:
        if j != i:
            neighbors[i].add(int(j))
            neighbors[int(j)].add(i)
degree = np.array([len(s) for s in neighbors])
print(f"Bậc trung bình: {degree.mean():.2f}, số nút cô lập: {(degree == 0).sum()}")

# 3. BFS từ nút có bậc lớn nhất cho đến khi đủ N nút
selected, visited = [], np.zeros(N_total, dtype=bool)
for start in np.argsort(-degree):
    if len(selected) >= args.num_nodes:
        break
    if visited[start]:
        continue
    queue = deque([int(start)])
    visited[start] = True
    while queue and len(selected) < args.num_nodes:
        u = queue.popleft()
        selected.append(u)
        for v in sorted(neighbors[u]):
            if not visited[v]:
                visited[v] = True
                queue.append(v)

idx = np.array(sorted(selected))
print(f"Đã chọn {len(idx)} nút")
np.save(os.path.join(out_dir, "node_ids.npy"), idx)

# 4. Cắt ma trận kề
sub_adj = np.asarray(adj[np.ix_(idx, idx)], dtype=np.float32)
np.save(os.path.join(out_dir, "adj_pems.npy"), sub_adj)
print(f"adj_pems.npy: {sub_adj.shape}, số cạnh khác 0: {(sub_adj > 0).sum()}")

# 5. Cắt dữ liệu train/val/test theo các nút đã chọn
for split in ["train", "val", "test"]:
    data = np.load(os.path.join(args.src, "short_term", f"{split}.npy"), mmap_mode="r")
    sub = np.asarray(data[:, idx, :], dtype=np.float32)
    np.save(os.path.join(out_dir, "short_term", f"{split}.npy"), sub)
    print(f"{split}.npy: {data.shape} -> {sub.shape}")

print(f"\nXong. Dữ liệu tập con nằm ở: {out_dir}")