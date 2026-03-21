import torch
from torch import nn
from torch.nn.parameter import Parameter


class ACSConv(nn.Module):
    def __init__(self, in_size, out_size, n_angles=8, K=8, bias=True):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.n_angles = n_angles
        self.K = K
        self.weight = Parameter(torch.Tensor(K, n_angles * in_size, out_size))

        nn.init.xavier_uniform_(self.weight)

        if bias:
            self.bias = Parameter(torch.Tensor(out_size))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)

    def forward(self, x, Ls):
        N = x.size(0)
        Tx_0 = x.repeat(self.n_angles, 1)

        feat_0 = (
            Tx_0
            .view(self.n_angles, N, self.in_size)
            .permute(1, 0, 2).contiguous()
            .view(N,self.n_angles * self.in_size)
        )

        out = torch.matmul(feat_0, self.weight[0])

        if self.K > 1:

            with torch.autocast(device_type='cuda', enabled=False):
                Tx_1_32 = torch.sparse.mm(Ls.float(), Tx_0.float())

            Tx_1 = Tx_1_32.to(Tx_0.dtype)

            feat_1 = (
                Tx_1
                .view(self.n_angles, N, self.in_size)
                .permute(1, 0, 2).contiguous()
                .view(N,self.n_angles * self.in_size)
            )

            out = out + torch.matmul(feat_1, self.weight[1])

        for k in range(2, self.K):

            with torch.autocast(device_type='cuda', enabled=False):
                Tx_2_32 = (
                    2.0 * torch.sparse.mm(Ls.float(), Tx_1.float())
                    - Tx_0.float()
                )

            Tx_2 = Tx_2_32.to(Tx_0.dtype)

            feat_2 = (
                Tx_2
                .view(self.n_angles, N, self.in_size)
                .permute(1, 0, 2).contiguous()
                .view(N,self.n_angles * self.in_size)
            )

            out = out + torch.matmul(feat_2, self.weight[k])

            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        return out