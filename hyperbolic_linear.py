import torch
import geoopt
from geoopt import ManifoldParameter


class HyperbolicLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, c=1.0, bias=True, init_std=1e-2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.ball = geoopt.PoincareBall(c=self.c)
        self.init_std = init_std

        # 权重可以直接用 random_normal（因为是二维）
        self.weight = ManifoldParameter(torch.randn((self.in_features, self.out_features)), manifold=self.ball)
        #  偏置不能用 random_normal（因为是一维），改用 expmap0 初始化
        if bias:
            bias_tangent = torch.randn(out_features) * self.init_std
            bias_init = self.ball.expmap0(bias_tangent)
            self.bias = ManifoldParameter(bias_init, manifold=self.ball)
        else:
            self.bias = None

    def forward(self, x):
        assert x.shape[-1] == self.in_features, f"Expected input features {self.in_features}, got {x.shape[-1]}"
        # 转置 weight：从 (out_features, in_features) -> (in_features, out_features)
        out = self.ball.mobius_matvec(self.weight.transpose(0, 1), x)
        if self.bias is not None:
            out = self.ball.mobius_add(out, self.bias)
        return out