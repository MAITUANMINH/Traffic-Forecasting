# Traffic-Forecasting: Tái hiện và cải tiến BigST

Báo cáo giữa kỳ và cuối kỳ môn Phân tích dữ liệu dự báo (cô Nguyễn Thị Hậu). Sinh viên: Mai Tuấn Minh – 240222691.

- **Giữa kỳ:** tái hiện bài báo và so sánh với 8 mô hình.
- **Cuối kỳ:** cải tiến BigST (nhánh đồ thị cục bộ, dự báo phần thay đổi) và thí nghiệm độ nhạy r, τ.

Bài báo:

> Jindong Han, Weijia Zhang, Hao Liu, Tao Tao, Naiqiang Tan, Hui Xiong. *BigST: Linear Complexity Spatio-Temporal Graph Neural Network for Traffic Forecasting on Large-Scale Road Networks.* PVLDB 17(5), 2024. DOI: 10.14778/3641204.3641217

Code gốc: https://github.com/usail-hkust/BigST (các file `model.py`, `trainer.py`, `run.py`, `util.py`, `metrics.py`, `preprocess/`). README gốc giữ tại `README_BigST.md.md`.

## Kết quả tái hiện (giữa kỳ)

Tập con 1.000 nút của bộ California (PeMS), dự báo 12 bước (60 phút), 3.227 cửa sổ test chung cho mọi mô hình. MAE, RMSE tính bằng dặm/giờ.

| Mô hình | MAE 15 phút | MAE 30 phút | MAE 60 phút | MAE TB | RMSE TB | MAPE TB |
|---|---|---|---|---|---|---|
| HA | 2,948 | 2,947 | 2,946 | 2,947 | 5,185 | 6,60% |
| HA 7 ngày | 2,666 | 2,665 | 2,664 | 2,665 | 4,885 | 6,03% |
| Persistence | 1,524 | 2,020 | 2,704 | 2,000 | 4,029 | 4,02% |
| VAR | 1,500 | 1,965 | 2,572 | 1,936 | 3,825 | 3,90% |
| LSTM | 1,336 | 1,704 | 2,148 | 1,668 | 3,468 | 3,52% |
| DCRNN | 1,316 | 1,706 | 2,201 | 1,678 | 3,506 | 3,47% |
| AGCRN | 1,301 | 1,589 | 1,852 | 1,535 | 3,279 | 3,29% |
| GWNET | **1,274** | 1,580 | 1,902 | 1,533 | 3,218 | 3,25% |
| **BigST** | 1,276 | **1,544** | **1,772** | **1,484** | **3,119** | **3,16%** |

- Thứ hạng của 6 mô hình chung với Bảng 2 của bài báo (BigST, GWNET, AGCRN, DCRNN, VAR, HA) khớp hoàn toàn.
- Loại bỏ thành phần: bỏ LSFE làm MAE tăng 1,8% (1,511); bỏ đặc trưng chu kỳ +0,1% (1,485); bỏ cả hai 1,529.
- Tính sẵn đặc trưng dài hạn: mỗi epoch giai đoạn 2 từ khoảng 60 phút còn 65 giây (nhanh khoảng 55 lần).
- Chi phí theo số nút: ma trận kề đầy đủ hết bộ nhớ GPU 8 GB từ 50.000 nút; tích chập tuyến tính của BigST cần 531 MB ở 100.000 nút; điểm giao khoảng 5.000 nút.

## Cải tiến BigST (cuối kỳ)

Mục 4.4 của bài báo thừa nhận phép xấp xỉ tích chập không gian *"leads to a noticeable performance degradation"*. Thí nghiệm giữa kỳ cho thấy BigST yếu ở dự báo ngắn hạn và làm mượt các cú sụt tốc độ. Hai cải tiến trong `model_cai_tien.py` (kế thừa `BigST`, không sửa `model.py` gốc):

- **A. Nhánh đồ thị cục bộ** (`--local_graph`): ở mỗi lớp lan truyền, cộng thêm phép lan truyền chính xác trên ma trận kề đường thật đã chuẩn hóa: `H(k) = LinearConv(H(k-1)) + A_hat · H(k-1) · W_loc(k)`. `A_hat` thưa (~6,7 cạnh/nút) nên mô hình vẫn tuyến tính theo N.
- **B. Dự báo phần thay đổi** (`--predict_delta`): `X_hat(t+h) = X(t) + Delta(h)`.

| Cấu hình | MAE 15 phút | MAE 30 phút | MAE 60 phút | MAE TB | RMSE TB | MAPE TB | Số tham số |
|---|---|---|---|---|---|---|---|
| BigST gốc | 1,276 | 1,544 | 1,772 | 1,484 | 3,119 | 3,16% | 152.284 |
| + A | 1,268 | 1,537 | 1,765 | 1,478 | 3,103 | 3,14% | 201.436 |
| + B | 1,273 | 1,540 | 1,770 | 1,479 | 3,104 | 3,14% | 152.284 |
| **+ A + B** | **1,265** | **1,530** | **1,758** | **1,470** | **3,077** | **3,12%** | 201.436 |
| GWNET | 1,274 | 1,580 | 1,902 | 1,533 | 3,218 | 3,25% | 316.812 |

- Kiểm tra lại với seed 1: BigST gốc 1,4842, A+B 1,4719. Trung bình 2 seed MAE giảm 0,87% (1,4840 → 1,4710), RMSE giảm 1,08%.
- Mốc 15 phút: A+B vượt GWNET (1,265 so với 1,274).
- Demo cảm biến 6072 lúc bắt đầu ùn tắc (05:25): MAE 60 phút BigST 13,46 → cải tiến 4,88 (GWNET 4,21); toàn mạng 1,58 → 1,52 (GWNET 1,74). Cảm biến 217 (06:55): 3,64 → 3,59; toàn mạng 2,52 → 2,40 (GWNET 2,51).
- Chi phí 1 lớp không gian ở 100.000 nút: BigST 15,2 ms / 531 MB, BigST + A 26,0 ms / 596 MB; ma trận kề đầy đủ hết bộ nhớ từ 50.000 nút.

### Độ nhạy r và τ (tái hiện mục 4.5, Figure 5–6 của bài báo)

| τ (r = 64) | 0,2 | 0,25 | 0,6 | 1,0 |
|---|---|---|---|---|
| MAE TB | 1,4843 | 1,4837 | 1,4881 | 1,4885 |

| r (τ = 0,2) | 16 | 32 | 64 |
|---|---|---|---|
| BigST gốc | 1,4913 | 1,4885 | 1,4843 |
| BigST + A + B | 1,4758 | – | 1,4701 |

- Xu hướng khớp bài báo: τ nhỏ và r lớn tốt hơn, biên độ chỉ ~0,005–0,007 MAE.
- BigST cải tiến với r = 16 vẫn tốt hơn BigST gốc với r = 64.

## Môi trường

- Windows, GPU NVIDIA RTX 4060 Laptop 8 GB
- Python 3.11.9, PyTorch (CUDA 12.6), numpy, scipy, pandas, matplotlib

```
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
python -m pip install numpy pandas scipy matplotlib
```

Trên PowerShell, để in tiếng Việt đúng: chạy `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8` một lần mỗi cửa sổ, và chạy Python với `python -X utf8 -u`.

## Dữ liệu

California (PeMS) do tác giả công bố (link trong `README_BigST.md.md`), giải nén vào `data/pems_data/pems_data/`:

```
data/pems_data/pems_data/
├── adj_pems.npy            (9638, 9638)
└── short_term/
    ├── train.npy           (18346, 9638, 3)   kênh: tốc độ, thời điểm trong ngày, ngày trong tuần
    ├── val.npy             (2620, 9638, 3)
    └── test.npy            (5242, 9638, 3)
```

Dữ liệu và checkpoint không đưa lên GitHub vì dung lượng lớn (xem `.gitignore`).

## Thứ tự chạy

| Bước | Lệnh | Kết quả |
|---|---|---|
| 0. Kiểm tra dữ liệu | `python inspect_data.py` | in shape các file |
| 1. Tạo tập con 1.000 nút liên thông (BFS) | `python make_subset.py --num_nodes 1000` | `data/pems_sub1000/` |
| 2. Thống kê dữ liệu | `python dataset_stats.py` | `results/dataset_stats.csv` |
| 3. Khám phá dữ liệu | `python eda.py` | `results/eda_*.png`, `eda_stats.csv` |
| 4. Giai đoạn 1: huấn luyện trước LSFE | `python preprocess/preprocess.py --data data/pems_sub1000/short_term --num_nodes 1000 --input_length 2016 --epochs 1 --print_every 500 --save checkpoint/` | `checkpoint/linear_transformer.pth` |
| 5. Tính sẵn đặc trưng dài hạn | `python precompute_features.py --data_path data/pems_sub1000/short_term --num_nodes 1000` | `data/pems_sub1000/features/` |
| 6. Giai đoạn 2: BigST đầy đủ | `python run_fast.py --use_long --name B_BigST --epochs 50` | `results/B_BigST.csv` |
| 7. Loại bỏ thành phần | `python run_fast.py --name A_khongDaiHan --epochs 50`<br>`python run_fast.py --use_long --feat_mode period_only --name B_woCLT --epochs 50`<br>`python run_fast.py --use_long --feat_mode long_only --name B_woPFS --epochs 50` | `results/*.csv` |
| 8. HA, HA 7 ngày, Persistence | `python baselines.py` | `results/HA.csv`, `HA_7ngay.csv`, `Persistence.csv` |
| 9. VAR | `python var_baseline.py --lags 6 --lambdas 1e5 1e6 1e7 1e8` | `results/VAR.csv` |
| 10. Graph WaveNet | `python gwnet_baseline.py --epochs 50` | `results/GWNET.csv` |
| 11. LSTM, AGCRN, DCRNN | `python deep_baselines.py --model lstm --epochs 30 --patience 10 --batch_size 32`<br>`python deep_baselines.py --model agcrn --epochs 30 --patience 10`<br>`python deep_baselines.py --model dcrnn --epochs 20 --patience 10 --learning_rate 0.003` | `results/LSTM.csv`, `AGCRN.csv`, `DCRNN.csv` |
| 12. Bảng tổng hợp, đối chiếu bài báo | `python summarize_results.py` | `results/tong_hop.csv`, `bang_ket_qua.tex`, `doi_chieu_bai_bao.csv` |
| 13. Chi phí theo số nút | `python benchmark_scaling.py` | `results/scaling.csv`, `scaling.png` |
| 14. Hình dự báo 2 cảm biến | `python plot_predictions.py` | `results/du_bao_nut_*.png` |
| 15. Demo | `python demo.py --sensor 217 --day 1 --time 06:55`<br>`python demo.py --sensor 6072 --day 2 --time 05:25` | in bảng dự báo, `results/demo_*.png` |
| 16. Kiểm tra code cải tiến | `python kiem_tra_cai_tien.py` | in kết quả kiểm tra |
| 17. Cải tiến A, B, A+B | `powershell -ExecutionPolicy Bypass -File chay_cai_tien.ps1`<br>`python run_fast.py --use_long --epochs 50 --seed 1 --name B_BigST_s1`<br>`python run_fast.py --use_long --epochs 50 --seed 1 --local_graph --predict_delta --name C_AB_s1` | `results/C_*.csv`, `B_BigST_s1.csv` |
| 18. Chi phí bản cải tiến theo số nút | `python benchmark_cai_tien.py` | `results/scaling_cai_tien.csv`, `.png` |
| 19. Độ nhạy r và τ | `powershell -ExecutionPolicy Bypass -File chay_do_nhay.ps1`<br>`python ve_do_nhay.py` | `results/S_*.csv`, `C_AB_r16.csv`, `do_nhay_r_tau.png` |

Log của từng lần chạy nằm trong `logs/`. `deep_baselines.py` lưu trạng thái sau mỗi epoch; thêm `--resume` để chạy tiếp nếu bị dừng.

## Các file được thêm vào

| File | Mô tả |
|---|---|
| `inspect_data.py` | Xem shape, kiểu dữ liệu các file `.npy` |
| `make_subset.py` | Cắt tập con N nút liên thông bằng BFS từ nút bậc lớn nhất |
| `dataset_stats.py` | Thống kê bộ đầy đủ và tập con: số nút, số bước, tỉ lệ thiếu, tốc độ, số cạnh |
| `eda.py` | Khám phá dữ liệu: phân bố tốc độ, chu kỳ ngày/tuần, tự tương quan, tương quan giữa cảm biến |
| `precompute_features.py` | Tính sẵn h_long (32 chiều, LSFE) và h_pe (12 chiều, chu kỳ 7 ngày) một lần, đúng mục 3.2.3 của bài báo |
| `run_fast.py` | Giai đoạn 2 đọc đặc trưng tính sẵn; `--feat_mode` cho loại bỏ thành phần |
| `baselines.py` | HA, HA 7 ngày, Persistence |
| `var_baseline.py` | VAR (hồi quy ridge đa biến, chọn λ theo tập val) |
| `gwnet_baseline.py` | Graph WaveNet cài lại theo mã gốc |
| `common.py` | Dữ liệu, hàm đo, vòng huấn luyện dùng chung cho các mô hình so sánh; hỗ trợ `--resume` |
| `deep_baselines.py` | LSTM, AGCRN, DCRNN cài lại theo mã gốc |
| `summarize_results.py` | Gom kết quả, xuất bảng LaTeX, đối chiếu thứ hạng với bài báo |
| `benchmark_scaling.py` | Thời gian và bộ nhớ của tích chập không gian đầy đủ so với tuyến tính, 1.000 → 100.000 nút |
| `plot_predictions.py` | Vẽ tốc độ thực tế và dự báo tại một cảm biến trong 2 ngày test |
| `demo.py` | Dự báo 60 phút tới tại một cảm biến bằng BigST, BigST cải tiến và GWNET đã huấn luyện |
| `model_cai_tien.py` | **Cuối kỳ.** BigST cải tiến: nhánh đồ thị cục bộ (`--local_graph`, `--local_sparse`) và dự báo phần thay đổi (`--predict_delta`) |
| `kiem_tra_cai_tien.py` | **Cuối kỳ.** Kiểm tra bản cải tiến: tắt cải tiến thì giống hệt bản gốc, đúng kích thước, ma trận dày và thưa cho cùng kết quả |
| `benchmark_cai_tien.py` | **Cuối kỳ.** Thời gian và bộ nhớ của ma trận đầy đủ, BigST và BigST + A, 1.000 → 100.000 nút |
| `chay_cai_tien.ps1`, `chay_do_nhay.ps1` | **Cuối kỳ.** Chạy liền các cấu hình cải tiến và độ nhạy |
| `ve_do_nhay.py` | **Cuối kỳ.** Vẽ hình độ nhạy r và τ |
| `fix_logs.py` | Chuyển log PowerShell bị lỗi mã hóa về UTF-8 |
| `approx_error.py` | Sai số xấp xỉ ma trận kề bằng đặc trưng ngẫu nhiên theo r và độ nhọn của phân bố |

## Các sửa đổi trên code gốc

1. `torch.qr` (đã bị xóa trong PyTorch mới) → `torch.linalg.qr` trong `model.py` và `preprocess/model.py`.
2. Bỏ tham số `verbose` của `MultiStepLR` (đã bị xóa trong PyTorch mới).
3. Thêm `import numpy as np` còn thiếu.

## Ghi chú

- DCRNN với tốc độ học 0,01 của mã gốc bị phân kỳ khi lô 16 (log `logs/dcrnn_lr0.01.txt`); bản chính dùng 0,003.
- Thời gian mỗi epoch của AGCRN đã loại 1 epoch bị gián đoạn do máy chuyển sang chế độ ngủ.
- Mỗi cấu hình chạy một seed (42), riêng BigST gốc và A+B chạy thêm seed 1.
- Thời gian mỗi epoch trên laptop dao động 40–82 giây tùy chế độ nguồn, nên so sánh chi phí dùng `benchmark_cai_tien.py` thay vì thời gian huấn luyện.
- Thí nghiệm cuối kỳ giữ tốc độ học 0,001 và τ = 0,25 theo mã nguồn (bài báo: 0,0001 và 0,2), trừ các lần chạy độ nhạy τ.
