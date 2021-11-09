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
         \ max(0, margin - x_i),  if y_i == -1
    """
    def __init__(self, margin: float = 1.0, reduction: str = "mean"):
        super(HingeEmbeddingLoss, self).__init__()
        self.loss = None
        self.margin = margin
        self.reduction = reduction

    def forward(self, x, y):
        if (y == 1.).all():
            self.loss = x
        if (y == -1.).all():
            self.loss = paddle.maximum(paddle.to_tensor(0.), self.margin - x)
        if self.reduction == 'mean':
            return self.loss.mean()
        if self.reduction == 'max':
            return self.loss.max()
        else:
            raise ValueError(f"choose reduction from ['mean', 'max'], but got {self.reduction}")
