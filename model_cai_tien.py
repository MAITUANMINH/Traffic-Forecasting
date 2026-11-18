
import torch
import torch.nn as nn

from model import BigST


class BigSTCaiTien(BigST):
    def __init__(self, args, supports=None, edge_indices=None):
        super().__init__(args, supports, edge_indices)
        self.local_graph = getattr(args, 'local_graph', False)
        self.predict_delta = getattr(args, 'predict_delta', False)
        self.local_sparse = getattr(args, 'local_sparse', False)

        if self.local_graph:
            if supports is None:
                raise ValueError('--local_graph cần ma trận kề (supports)')
            A = supports[0].float()                       # (N, N), đã chuẩn hóa D^-1 A
            if self.local_sparse:
                A = A.to_sparse()                         # dùng khi N lớn
            self.register_buffer('A_hat', A, persistent=False)
            dim = args.hid_dim * 4
            # mỗi lớp một ma trận W_loc riêng (Conv2d 1x1 = nhân ma trận theo chiều đặc trưng)
            self.local_fc = nn.ModuleList(
                [nn.Conv2d(dim, dim, kernel_size=(1, 1), bias=False) for _ in range(self.num_layers)])

    def lan_truyen_cuc_bo(self, h, i):
        """h: (B, C, N, 1) -> A_hat · h · W_loc, cùng kích thước."""
        h = self.local_fc[i](h)                           # (B, C, N, 1)
        if self.local_sparse:
            B, C, N, _ = h.shape
            h2 = h.squeeze(-1).permute(2, 0, 1).reshape(N, B * C)       # (N, B*C)
            h2 = torch.sparse.mm(self.A_hat, h2)                          # (N, B*C)
            return h2.reshape(N, B, C).permute(1, 2, 0).unsqueeze(-1)
        return torch.einsum('mn,bcnl->bcml', self.A_hat, h)

    def forward(self, x, feat=None):
        # input: (B, N, T, D)
        B, N, T, D = x.size()
        toc_do_hien_tai = x[:, :, -1, 0]                   # (B, N), đã chuẩn hóa

        time_emb = self.time_emb_layer[(x[:, :, -1, 1] * self.time_num).type(torch.LongTensor)]
        week_emb = self.week_emb_layer[(x[:, :, -1, 2]).type(torch.LongTensor)]

        x = x.contiguous().view(B, N, -1).transpose(1, 2).unsqueeze(-1)  # (B, D*T, N, 1)
        input_emb = self.input_emb_layer(x)

        node_emb = self.node_emb_layer.unsqueeze(0).expand(B, -1, -1).transpose(1, 2).unsqueeze(-1)
        time_emb = time_emb.transpose(1, 2).unsqueeze(-1)
        week_emb = week_emb.transpose(1, 2).unsqueeze(-1)

        x_g = torch.cat([node_emb, time_emb, week_emb], dim=1)
        x = torch.cat([input_emb, node_emb, time_emb, week_emb], dim=1)  # (B, dim*4, N, 1)

        x_pool = [x]
        node_vec1 = self.W_1(x_g).permute(0, 2, 3, 1)      # (B, N, 1, dim)
        node_vec2 = self.W_2(x_g).permute(0, 2, 3, 1)
        for i in range(self.num_layers):
            dau_vao = x
            if self.use_residual:
                residual = x
            x, node_vec1_prime, node_vec2_prime = self.linear_conv[i](x, node_vec1, node_vec2)

            # ---- CẢI TIẾN A: cộng nhánh cục bộ chính xác ----
            if self.local_graph:
                x = x + self.lan_truyen_cuc_bo(dau_vao, i)

            if self.use_residual:
                x = x + residual
            if self.use_bn:
                x = x.permute(0, 2, 3, 1)
                x = self.bn[i](x)
                x = x.permute(0, 3, 1, 2)

        x_pool.append(x)
        x = torch.cat(x_pool, dim=1)
        x = self.activation(x)

        if self.use_long:
            feat = feat.permute(0, 2, 1).unsqueeze(-1)
            x = torch.cat([x, feat], dim=1)
        x = self.regression_layer(x)                       # (B, T_out, N, 1)
        x = x.squeeze(-1).permute(0, 2, 1)                 # (B, N, T_out)

        # ---- CẢI TIẾN B: dự báo phần thay đổi ----
        if self.predict_delta:
            x = x + toc_do_hien_tai.unsqueeze(-1)

        if self.use_spatial:
            from model import spatial_loss
            s_loss = spatial_loss(node_vec1_prime, node_vec2_prime, self.supports, self.edge_indices)
            return x, s_loss
        return x, 0