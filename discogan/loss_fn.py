# encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
from paddle import nn


class HingeEmbeddingLoss(nn.Layer):
    """
         / x_i,                   if y_i == 1
    l_i =
         \ max(0, delta - x_i),  if y_i == -1
    """

    def __init__(self, delta: float = 1.0, reduction: str = "mean"):
        super(HingeEmbeddingLoss, self).__init__()
        self.loss = None
        self.delta = delta
        self.reduction = reduction

        if self.reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                "'reduction' in 'hinge_embedding_loss' should be 'sum', 'mean' or 'none', "
                "but received {}.".format(self.reduction))

    def forward(self, x, y):
        if set(y.flatten().numpy()) <= {1., -1.}:
            self.loss = paddle.where(
                y == 1., x,
                paddle.maximum(paddle.to_tensor(0.), self.delta - x))
        else:
            raise ValueError("'label' should contain 1. or -1., "
                             "but received label containing {}.".format(
                             set(y.flatten().numpy())))

        if self.reduction == 'mean':
            return paddle.mean(self.loss)
        elif self.reduction == 'sum':
            return paddle.sum(self.loss)
        elif self.reduction == 'none':
            return self.loss
