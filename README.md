# Traffic-Forecasting: Tái hiện và mở rộng BigST

Báo cáo giữa kỳ môn Phân tích dữ liệu dự báo. Tái hiện bài báo:

> Jindong Han, Weijia Zhang, Hao Liu, Tao Tao, Naiqiang Tan, Hui Xiong. *BigST: Linear Complexity Spatio-Temporal Graph Neural Network for Traffic Forecasting on Large-Scale Road Networks.* PVLDB 17(5), 2024. DOI: 10.14778/3641204.3641217

Code gốc: https://github.com/usail-hkust/BigST (các file `model.py`, `trainer.py`, `run.py`, `util.py`, `metrics.py`, `preprocess/`). README gốc giữ tại `README_BigST.md`.

## Môi trường

- Windows, GPU NVIDIA RTX 4060 Laptop 8GB
- Python 3.11.9, PyTorch (CUDA 12.6), numpy, scipy, pandas, matplotlib

```
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
python -m pip install numpy pandas scipy matplotlib
```

Trên PowerShell 5.1, để in tiếng Việt đúng: chạy `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8` một lần mỗi cửa sổ, và chạy Python với `python -X utf8 -u`.

## Dữ liệu

California (PeMS) do tác giả công bố (link trong `README_BigST.md`), giải nén vào `data/pems_data/pems_data/`:

```
data/pems_data/pems_data/
├── adj_pems.npy            (9638, 9638)
└── short_term/
    ├── train.npy           (18346, 9638, 3)   kênh: tốc độ, thời điểm trong ngày, ngày trong tuần
    ├── val.npy             (2620, 9638, 3)
    └── test.npy            (5242, 9638, 3)
```

Dữ liệu và checkpoint không đưa lên GitHub (xem `.gitignore`).

## Thứ tự chạy

| Bước | Lệnh | Kết quả |
|---|---|---|
| 0. Kiểm tra dữ liệu | `python inspect_data.py` | in shape các file |
| 1. Tạo tập con 1.000 nút liên thông (BFS) | `python make_subset.py --num_nodes 1000` | `data/pems_sub1000/` |
| 2. Thống kê dữ liệu (Bảng 1) | `python dataset_stats.py` | `results/dataset_stats.csv` |
| 3. Giai đoạn 1: huấn luyện trước LSFE | `python preprocess/preprocess.py --data data/pems_sub1000/short_term --num_nodes 1000 --input_length 2016 --epochs 1 --print_every 500 --save checkpoint/` | `checkpoint/linear_transformer.pth` |
| 4. Tính sẵn đặc trưng dài hạn | `python precompute_features.py --data_path data/pems_sub1000/short_term --num_nodes 1000` | `data/pems_sub1000/features/` |
| 5. Giai đoạn 2: BigST đầy đủ | `python run_fast.py --use_long --name B_BigST --epochs 50` | `results/B_BigST.csv` |
| 6. Ablation | `python run_fast.py --name A_khongDaiHan --epochs 50`<br>`python run_fast.py --use_long --feat_mode period_only --name B_woCLT --epochs 50`<br>`python run_fast.py --use_long --feat_mode long_only --name B_woPFS --epochs 50` | `results/*.csv` |
| 7. Baseline đơn giản | `python baselines.py` | `results/HA.csv`, `HA_7ngay.csv`, `Persistence.csv` |
| 8. Baseline GWNET | `python gwnet_baseline.py --epochs 50` | `results/GWNET.csv` |
| 9. Bảng tổng hợp | `python summarize_results.py` | `results/tong_hop.csv`, `results/bang_ket_qua.tex` |
| 10. Mở rộng: độ phức tạp theo số nút | `python benchmark_scaling.py` | `results/scaling.csv`, `scaling.png` |
| 11. Mở rộng: sai số xấp xỉ theo r | `python approx_error.py` | `results/approx_error.csv`, `approx_error.png` |

Log của từng lần chạy nằm trong `logs/`.

## Các file code được thêm vào 

| File | Mô tả |
|---|---|
| `inspect_data.py` | Xem shape, kiểu dữ liệu các file `.npy` (không nạp hết vào RAM) |
| `make_subset.py` | Cắt tập con N nút liên thông bằng BFS từ nút bậc lớn nhất, lưu float32 |
| `dataset_stats.py` | Thống kê dữ liệu cho bộ đầy đủ và tập con |
| `precompute_features.py` | Tính sẵn h_long (32 chiều, LSFE) và h_pe (12 chiều, đặc trưng chu kỳ 7 ngày) cho mọi cửa sổ — đúng thiết kế mục 3.2.3 của bài báo |
| `run_fast.py` | Huấn luyện Giai đoạn 2 với đặc trưng tính sẵn; cùng tập mẫu (`--min_t 2015`) cho mọi cấu hình; `--feat_mode` cho ablation |
| `baselines.py` | HA, HA 7 ngày, Persistence trên cùng tập test và cùng cách đo |
| `gwnet_baseline.py` | Graph WaveNet, cùng dữ liệu, tập mẫu và cách đo |
| `summarize_results.py` | Gom kết quả thành bảng CSV và bảng LaTeX |
| `benchmark_scaling.py` | Thời gian và bộ nhớ của tích chập không gian dense N×N so với tuyến tính, N = 1.000 → 100.000 |
| `approx_error.py` | Sai số xấp xỉ ma trận kề bằng đặc trưng ngẫu nhiên theo r |

## Các sửa đổi trên code gốc

1. `torch.qr` (đã bị xóa trong PyTorch mới) → `torch.linalg.qr` trong `model.py` và `preprocess/model.py`.
2. Bỏ tham số `verbose` của `MultiStepLR` trong `trainer.py` (đã bị xóa trong PyTorch mới).
3. Thêm `import numpy as np` vào `model.py` và `preprocess/model.py`.

## Kết quả chính (tập con 1.000 nút, dự báo 12 bước = 1 giờ, 3.227 cửa sổ test)

| Mô hình | MAE h3 | MAE h6 | MAE h12 | MAE TB | MAPE TB | RMSE TB |
|---|---|---|---|---|---|---|
| HA | 2.948 | 2.947 | 2.946 | 2.947 | 6.60% | 5.185 |
| HA 7 ngày | 2.666 | 2.665 | 2.664 | 2.665 | 6.03% | 4.885 |
| Persistence | 1.524 | 2.020 | 2.704 | 2.000 | 4.02% | 4.029 |
| GWNET | **1.274** | 1.580 | 1.902 | 1.533 | 3.25% | 3.218 |
| BigST bỏ LSFE và PFS | 1.292 | 1.587 | 1.860 | 1.529 | 3.25% | 3.207 |
| BigST bỏ LSFE (woCLT) | 1.284 | 1.568 | 1.825 | 1.511 | 3.21% | 3.164 |
| BigST bỏ PFS (woPFS) | 1.276 | 1.548 | 1.775 | 1.485 | **3.14%** | 3.126 |
| BigST (đầy đủ) | 1.276 | **1.544** | **1.772** | **1.484** | 3.16% | **3.119** |

Chi tiết theo từng horizon trong `results/`.