
import argparse
import math

import numpy as np
import torch
import torch.nn as nn

from common import L_IN, L_OUT, TrafficData, load_adj, random_walk, train_and_test

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, choices=['lstm', 'agcrn', 'dcrnn'])
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--data_path', type=str, default='data/pems_sub1000/short_term')
parser.add_argument('--adjdata', type=str, default='data/pems_sub1000/adj_pems.npy')
parser.add_argument('--name', type=str, default=None, help='mặc định: tên mô hình viết hoa')
parser.add_argument('--min_t', type=int, default=2015)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--patience', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=None, help='mặc định theo từng mô hình')
parser.add_argument('--learning_rate', type=float, default=None, help='mặc định theo từng mô hình')
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--quick', action='store_true', help='chạy thử: 2 epoch, 20 batch mỗi tập, lưu tên *_thu')
parser.add_argument('--resume', action='store_true',
                    help='chạy tiếp từ checkpoint/<name>_dangchay.pth (lưu sau mỗi epoch)')
args = parser.parse_args()

# siêu tham số mặc định theo code gốc của từng mô hình (batch giảm cho vừa GPU 8 GB)
DEFAULTS = {'lstm': dict(bs=64, lr=0.001, in_dim=2),
            'agcrn': dict(bs=32, lr=0.003, in_dim=1),   # code gốc: batch 64, lr 0.003, chỉ kênh tốc độ
            'dcrnn': dict(bs=16, lr=0.01, in_dim=2)}    # code gốc: batch 64, lr 0.01, tốc độ + thời điểm
cfg = DEFAULTS[args.model]
args.batch_size = args.batch_size or cfg['bs']
args.learning_rate = args.learning_rate or cfg['lr']
args.name = args.name or args.model.upper()
if args.quick:
    args.epochs, args.name = 2, args.name + '_thu'
IN_DIM = cfg['in_dim']
print(args)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
data = TrafficData(args.data_path, args.min_t)
N = data.N


# ================================ LSTM ================================
class LSTMModel(nn.Module):
    """Mỗi cảm biến là một chuỗi riêng, dùng chung trọng số; không trao đổi thông tin giữa các nút."""

    def __init__(self, in_dim, hidden, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True, dropout=0.1)
        self.out = nn.Linear(hidden, L_OUT)

    def forward(self, x):                                    # x: (B, 12, N, C)
        B, T, Nn, C = x.shape
        z = x.permute(0, 2, 1, 3).reshape(B * Nn, T, C)      # (B*N, 12, C)
        h, _ = self.lstm(z)
        y = self.out(h[:, -1])                               # trạng thái ẩn cuối -> 12 bước
        return y.reshape(B, Nn, L_OUT).transpose(1, 2)       # (B, 12, N)


# ================================ AGCRN ================================
class AVWGCN(nn.Module):
    """Tích chập đồ thị với tham số riêng theo nút (NAPL) trên đồ thị học từ embedding (DAGG)."""

    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super().__init__()
        self.weights_pool = nn.Parameter(torch.empty(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.empty(embed_dim, dim_out))

    def node_params(self, E):
        """Trọng số riêng cho từng nút: W (N, K, C_in, C_out), b (N, C_out).
        Tính MỘT lần mỗi lượt chạy rồi dùng lại ở cả 12 bước để tiết kiệm bộ nhớ GPU
        (code gốc tính lại ở mỗi bước, kết quả giống hệt)."""
        return torch.einsum('nd,dkio->nkio', E, self.weights_pool), E @ self.bias_pool

    def forward(self, x, sup, W, b):                         # x: (B, N, C_in), sup: (K, N, N)
        xg = torch.einsum('knm,bmc->bknc', sup, x).permute(0, 2, 1, 3)   # (B, N, K, C_in)
        return torch.einsum('bnki,nkio->bno', xg, W) + b


class AGCRNCell(nn.Module):
    def __init__(self, dim_in, hidden, cheb_k, embed_dim):
        super().__init__()
        self.hidden = hidden
        self.gate = AVWGCN(dim_in + hidden, 2 * hidden, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in + hidden, hidden, cheb_k, embed_dim)

    def forward(self, x, h, sup, pg, pu):
        zr = torch.sigmoid(self.gate(torch.cat([x, h], -1), sup, *pg))
        z, r = torch.split(zr, self.hidden, dim=-1)
        hc = torch.tanh(self.update(torch.cat([x, z * h], -1), sup, *pu))
        return r * h + (1 - r) * hc


class AGCRN(nn.Module):
    def __init__(self, num_nodes, in_dim, hidden=64, embed_dim=10, cheb_k=2, layers=2):
        super().__init__()
        self.hidden, self.cheb_k = hidden, cheb_k
        self.E = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.cells = nn.ModuleList([AGCRNCell(in_dim if i == 0 else hidden, hidden, cheb_k, embed_dim)
                                    for i in range(layers)])
        self.end_conv = nn.Conv2d(1, L_OUT, kernel_size=(1, hidden))

    def supports(self):
        """Ma trận kề học được S = softmax(ReLU(E E^T)), kích thước N x N -> chi phí O(N^2)."""
        Nn = self.E.shape[0]
        S = torch.softmax(torch.relu(self.E @ self.E.T), dim=1)
        sup = [torch.eye(Nn, device=S.device), S]
        for _ in range(2, self.cheb_k):
            sup.append(2 * S @ sup[-1] - sup[-2])
        return torch.stack(sup)                              # (K, N, N)

    def forward(self, x):                                    # x: (B, 12, N, C)
        B, T, Nn, _ = x.shape
        sup = self.supports()
        seq = x
        for cell in self.cells:
            pg, pu = cell.gate.node_params(self.E), cell.update.node_params(self.E)
            h = torch.zeros(B, Nn, self.hidden, device=x.device)
            outs = []
            for t in range(T):
                h = cell(seq[:, t], h, sup, pg, pu)
                outs.append(h)
            seq = torch.stack(outs, 1)                       # (B, 12, N, H)
        last = seq[:, -1:]                                   # (B, 1, N, H)
        return self.end_conv(last).squeeze(-1)               # (B, 12, N)


# ================================ DCRNN ================================
class DiffusionConv(nn.Module):
    """Tích chập khuếch tán bậc K trên ma trận chuyển tiếp xuôi và ngược (đồ thị cố định)."""

    def __init__(self, supports, dim_in, dim_out, K=2):
        super().__init__()
        self.supports, self.K = supports, K
        n_mat = 1 + len(supports) * K
        self.lin = nn.Linear(dim_in * n_mat, dim_out)

    def forward(self, x):                                    # x: (B, N, C)
        xs = [x]
        for A in self.supports:
            x0, x1 = x, torch.einsum('nm,bmc->bnc', A, x)
            xs.append(x1)
            for _ in range(2, self.K + 1):
                x2 = 2 * torch.einsum('nm,bmc->bnc', A, x1) - x0
                xs.append(x2)
                x0, x1 = x1, x2
        return self.lin(torch.cat(xs, -1))


class DCGRUCell(nn.Module):
    def __init__(self, supports, dim_in, hidden, K=2):
        super().__init__()
        self.hidden = hidden
        self.gates = DiffusionConv(supports, dim_in + hidden, 2 * hidden, K)
        self.cand = DiffusionConv(supports, dim_in + hidden, hidden, K)

    def forward(self, x, h):
        ru = torch.sigmoid(self.gates(torch.cat([x, h], -1)))
        r, u = torch.split(ru, self.hidden, dim=-1)
        c = torch.tanh(self.cand(torch.cat([x, r * h], -1)))
        return u * h + (1 - u) * c


class DCRNN(nn.Module):
    def __init__(self, supports, in_dim, hidden=64, layers=2, K=2, cl_decay_steps=2000):
        super().__init__()
        self.hidden, self.cl = hidden, cl_decay_steps
        self.enc = nn.ModuleList([DCGRUCell(supports, in_dim if i == 0 else hidden, hidden, K) for i in range(layers)])
        self.dec = nn.ModuleList([DCGRUCell(supports, 1 if i == 0 else hidden, hidden, K) for i in range(layers)])
        self.proj = nn.Linear(hidden, 1)

    def forward(self, x, y_norm=None, step=0):               # x: (B, 12, N, C); y_norm: (B, 12, N)
        B, T, Nn, _ = x.shape
        hs = [torch.zeros(B, Nn, self.hidden, device=x.device) for _ in self.enc]
        for t in range(T):                                   # mã hóa 12 bước quá khứ
            inp = x[:, t]
            for i, cell in enumerate(self.enc):
                hs[i] = cell(inp, hs[i])
                inp = hs[i]
        go = torch.zeros(B, Nn, 1, device=x.device)          # ký hiệu bắt đầu giải mã
        outs = []
        # scheduled sampling: xác suất dùng nhãn thật giảm dần theo số bước huấn luyện
        p_true = self.cl / (self.cl + math.exp(step / self.cl)) if self.training and y_norm is not None else 0.0
        for t in range(L_OUT):                               # giải mã 12 bước tương lai
            inp = go
            for i, cell in enumerate(self.dec):
                hs[i] = cell(inp, hs[i])
                inp = hs[i]
            out = self.proj(inp)                             # (B, N, 1)
            outs.append(out.squeeze(-1))
            use_true = p_true > 0 and np.random.rand() < p_true
            go = y_norm[:, t].unsqueeze(-1) if use_true else out
        return torch.stack(outs, 1)                          # (B, 12, N)


# ================================ Chạy ================================
if args.model == 'lstm':
    model = LSTMModel(IN_DIM, args.hidden).to(dev)
    forward = lambda m, x, y, step, train: m(x)
    milestones = None
elif args.model == 'agcrn':
    model = AGCRN(N, IN_DIM, hidden=args.hidden).to(dev)
    for p in model.parameters():                             # khởi tạo như code gốc AGCRN
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    forward = lambda m, x, y, step, train: m(x)
    milestones = None
else:
    adj = load_adj(args.adjdata)
    supports = [torch.tensor(random_walk(adj), device=dev), torch.tensor(random_walk(adj.T), device=dev)]
    model = DCRNN(supports, IN_DIM, hidden=args.hidden).to(dev)
    forward = lambda m, x, y, step, train: m(x, y if train else None, step)
    milestones = [20, 30, 40]                                # code gốc: giảm lr 10 lần ở các epoch này

train_and_test(model, forward, data, args, dev, IN_DIM, milestones)