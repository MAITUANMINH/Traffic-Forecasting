"""
Gom các file results/<tên>.csv thành bảng so sánh theo horizon 3, 6, 12 và trung bình.
Xuất: results/tong_hop.csv (để xem) và results/bang_ket_qua.tex (bảng LaTeX dùng booktabs,
chép vào báo cáo; nhớ \\usepackage{booktabs, multirow}). Giá trị tốt nhất mỗi cột được in đậm.

Ví dụ:
    python summarize_results.py
"""
import csv
import os

# (tên file trong results/, tên hiển thị) - file nào chưa có sẽ tự bỏ qua
MODELS = [
    ('HA', 'HA'),
    ('HA_7ngay', 'HA 7 ngày'),
    ('Persistence', 'Persistence'),
    ('GWNET', 'GWNET'),
    ('A_khongDaiHan', 'BigST bỏ LSFE và PFS'),
    ('B_woCLT', 'BigST bỏ LSFE (woCLT)'),
    ('B_woPFS', 'BigST bỏ PFS (woPFS)'),
    ('B_BigST', 'BigST (đầy đủ)'),
]
HORIZONS = [('3', '15 phút (h3)'), ('6', '30 phút (h6)'), ('12', '60 phút (h12)'), ('avg', 'Trung bình')]


def load(name):
    path = os.path.join('results', f'{name}.csv')
    if not os.path.exists(path):
        return None
    out = {}
    with open(path, encoding='utf-8') as f:
        for row in csv.reader(f):
            if len(row) == 4 and row[0] != 'horizon':
                out[row[0]] = [float(v) for v in row[1:]]
    return out


rows = [(label, load(name)) for name, label in MODELS]
rows = [(label, r) for label, r in rows if r is not None]
print(f'Tìm thấy {len(rows)} mô hình: ' + ', '.join(l for l, _ in rows))

# giá trị tốt nhất (nhỏ nhất) mỗi cột
best = {(h, m): min(r[h][m] for _, r in rows) for h, _ in HORIZONS for m in range(3)}

with open('results/tong_hop.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['Mô hình'] + [f'{t} {m}' for _, t in HORIZONS for m in ('MAE', 'MAPE(%)', 'RMSE')])
    for label, r in rows:
        w.writerow([label] + [f'{r[h][m] * (100 if m == 1 else 1):.2f}' for h, _ in HORIZONS for m in range(3)])


def cell(v, is_best, pct=False):
    s = f'{v * 100:.2f}\\%' if pct else f'{v:.2f}'
    return f'\\textbf{{{s}}}' if is_best else s


lines = [
    '\\begin{table}[htbp]',
    '\\centering',
    '\\caption{So sánh kết quả dự báo trên tập con 1.000 nút của bộ dữ liệu California}',
    '\\label{tab:ketqua}',
    '\\resizebox{\\textwidth}{!}{%',
    '\\begin{tabular}{l' + 'ccc' * len(HORIZONS) + '}',
    '\\toprule',
    '\\multirow{2}{*}{Mô hình} & ' + ' & '.join(f'\\multicolumn{{3}}{{c}}{{{t}}}' for _, t in HORIZONS) + ' \\\\',
    ' '.join(f'\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}' for i in range(len(HORIZONS))),
    ' & ' + ' & '.join(['MAE & MAPE & RMSE'] * len(HORIZONS)) + ' \\\\',
    '\\midrule',
]
for label, r in rows:
    cells = [cell(r[h][m], abs(r[h][m] - best[(h, m)]) < 1e-9, pct=(m == 1)) for h, _ in HORIZONS for m in range(3)]
    lines.append(f'{label} & ' + ' & '.join(cells) + ' \\\\')
lines += ['\\bottomrule', '\\end{tabular}}', '\\end{table}']
with open('results/bang_ket_qua.tex', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')

print('\n' + f'{"Mô hình":26s}' + ''.join(f'{t:>24s}' for _, t in HORIZONS))
for label, r in rows:
    print(f'{label:26s}' + ''.join(f'{r[h][0]:8.3f}{r[h][1] * 100:7.2f}%{r[h][2]:8.3f}' for h, _ in HORIZONS))
print('\n(mỗi nhóm: MAE, MAPE, RMSE)')
print('Đã lưu: results/tong_hop.csv, results/bang_ket_qua.tex')