# ======================================================================
# Copyright (c) 2023 Fan Yang
# Robotic Systems Lab, ETH Zurich
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ======================================================================

import torch
torch.set_default_dtype(torch.float32)

class CubicSplineTorch:
    # Reference: https://stackoverflow.com/questions/61616810/how-to-do-cubic-spline-interpolation-and-integration-in-pytorch
    def __init__(self):
        return None

    def h_poly(self, t):
        alpha = torch.arange(4, device=t.device, dtype=t.dtype)
        tt = t[:, None, :] ** alpha[None, :, None]
        A = torch.tensor([[1, 0, -3, 2], [0, 1, -2, 1], [0, 0, 3, -2], [0, 0, -1, 1]], dtype=t.dtype, device=t.device)
        return A @ tt

    def interp(self, x, y, xs):
        m = (y[:, 1:, :] - y[:, :-1, :]) / torch.unsqueeze(x[:, 1:] - x[:, :-1], 2)
        m = torch.cat([m[:, None, 0], (m[:, 1:] + m[:, :-1]) / 2, m[:, None, -1]], 1)
        idxs = torch.searchsorted(x[0, 1:], xs[0, :])
        dx = x[:, idxs + 1] - x[:, idxs]
        hh = self.h_poly((xs - x[:, idxs]) / dx)
        hh = torch.transpose(hh, 1, 2)
        out = hh[:, :, 0:1] * y[:, idxs, :]
        out = out + hh[:, :, 1:2] * m[:, idxs] * dx[:, :, None]
        out = out + hh[:, :, 2:3] * y[:, idxs + 1, :]
        out = out + hh[:, :, 3:4] * m[:, idxs + 1] * dx[:, :, None]
        return out


class TrajOpt:
    def __init__(self):
        self.cs_interp = CubicSplineTorch()
        return None

    def TrajGeneratorFromPFreeRot(self, preds, step):
        # Points is in se3
        batch_size, num_p, dims = preds.shape
        points_preds = torch.cat(
            (torch.zeros(batch_size, 1, dims, device=preds.device, requires_grad=preds.requires_grad), preds), axis=1
        )
        num_p = num_p + 1
        xs = torch.arange(0, num_p - 1 + step, step, device=preds.device)
        xs = xs.repeat(batch_size, 1)
        x = torch.arange(num_p, device=preds.device, dtype=preds.dtype)
        x = x.repeat(batch_size, 1)
        waypoints = self.cs_interp.interp(x, points_preds, xs)
        return waypoints  # R3
