# MLP Reconstruction 实施指导（纯推理加速，不涉及量化）

> 本指导基于 CVPR 2025 论文 APHQ-ViT，提取其中 MLP Reconstruction 部分，
> 目标是将 MobileCLIP2-B 中 MLP 的 GELU 替换为 ReLU，并通过知识蒸馏恢复精度。
> **不涉及任何量化操作。**

---

## 一、背景与目标

### 做什么

将模型 MLP 中的 GELU 激活函数替换为 ReLU，同时通过蒸馏损失让 ReLU-MLP 的输出尽量逼近原始 GELU-MLP，保持模型精度。

### 为什么能这样做

- 深层 Transformer 训练时用 GELU 是为了避免 dying ReLU 问题
- **但 MLP Reconstruction 是逐层单独重建的（shallow depth）**，不存在 dying ReLU 问题
- ReLU 网络理论上同样具有通用近似能力（universal approximation）
- ReLU 可以 fold 进前一个线性层，推理更快

### 收益

- 推理速度提升约 1.1～1.2x（ReLU 比 GELU 计算更快，且可 kernel fusion）
- 精度损失极小（论文消融实验显示多数模型精度损失 < 0.5%）

---

## 二、整体流程

```
对每个 Transformer Block 中的 MLP，依次执行：

Step 1  用标定数据前向，收集原始 GELU-MLP 的输出 O_GELU
Step 2  计算 APH 权重 H_bar（衡量每个输出维度的重要性）
Step 3  将 MLP 的 GELU 替换为 ReLU
Step 4  用 L_Direct + L_Clamp 蒸馏损失训练 ReLU-MLP，使其逼近 O_GELU
Step 5  保存重建后的 MLP 权重，替换原模型
```

**注意：逐层串行处理，每次只重建当前层，其余层保持原样。**

---

## 三、各步骤详细说明

### Step 1：收集 GELU-MLP 的原始输出

对标定集（calibration set）中每条样本，前向传播到当前 MLP 层，记录：

- 输入 `X`（MLP 的输入）
- 输出 `O_GELU`（GELU 激活的 MLP 输出）

这两者在后续蒸馏中作为 teacher 信号，**只需收集一次，不需要梯度**。

---

### Step 2：计算 APH 权重 H_bar

APH（Average Perturbation Hessian）用于给输出的每个维度赋予重要性权重，让损失函数更关注对最终任务影响大的维度。

**计算方法：**

对每条标定样本 n，对 MLP 的输出 `O` 施加微小正负扰动：

```
O_plus  = O + delta     # delta = 1e-6
O_minus = O - delta
```

将 `O_plus` 和 `O_minus` 分别继续前向传播到模型末尾，计算蒸馏 loss，再反向传播，得到各自的梯度（Jacobian）：

```
J_plus  = ∂L / ∂O  evaluated at O_plus
J_minus = ∂L / ∂O  evaluated at O_minus
```

对角 Hessian 近似：

```
H_n[i] = (J_plus[i] - J_minus[i]) / (2 * delta)
```

对所有标定样本取均值：

```
H_bar[i] = mean over n of H_n[i]
```

`H_bar` 的 shape 与 MLP 输出维度相同，作为 element-wise 权重使用。

**实现提示：**

- `delta = 1e-6`
- 蒸馏 loss 对分类任务用 KL Divergence（logits 之间），对检索任务可用 cosine similarity loss 或 L2
- APH 计算完后固定不变，仅用于加权

---

### Step 3：替换 GELU 为 ReLU

将当前 MLP block 中所有 GELU 替换为 ReLU。其他层保持不变。

```python
# 伪代码
for module in mlp_block.modules():
    if isinstance(module, nn.GELU):
        # 替换为 ReLU（in-place）
        replace_with_relu(module)
```

此时 ReLU-MLP 的权重与原始相同，输出会与 O_GELU 有偏差，需要通过蒸馏纠正。

---

### Step 4：蒸馏训练（核心）

对当前 MLP 进行蒸馏重建，让 ReLU-MLP 的输出逼近 O_GELU。

#### 4.1 Direct Loss

直接比较 ReLU-MLP 输出与 GELU-MLP 输出的加权 L2：

```
O_Direct = FC2(ReLU(FC1(X)))

L_Direct = sum_i [ (O_GELU[i] - O_Direct[i])^2 * H_bar[i] ]
```

#### 4.2 Clamp Loss

对 ReLU 后的中间激活进行截断，模拟量化（或部署时）的值域约束：

```
A_FC2 = ReLU(FC1(X))                         # ReLU 后的中间激活
threshold = quantile(A_FC2_positive, p=0.99) # 取所有正值的 99 百分位
A_clamped = clamp(A_FC2, max=threshold)       # 截断超出部分
O_clamp = FC2(A_clamped)

L_Clamp = sum_i [ (O_GELU[i] - O_clamp[i])^2 * H_bar[i] ]
```

**注意：** `threshold` 在每个 batch 内基于当前 batch 的激活值动态计算，不是全局固定的。

#### 4.3 总蒸馏 Loss

```
L_Distill = L_Direct + alpha * L_Clamp
alpha = 2  （固定超参）
```

**⚠️ 关键：L_Direct 不可省略。**
单独使用 L_Clamp 时，被 hard-clip 的区域梯度为零，导致梯度消失，参数无法更新。L_Direct 保留了未截断激活的梯度通道，缓解这一问题。

#### 4.4 训练设置

| 超参数         | 值                                     |
| -------------- | -------------------------------------- |
| batch size     | 32                                     |
| learning rate  | 1e-3（权重），4e-5（如有可学习 scale） |
| max iterations | 20000                                  |
| clamp 百分位 p | 0.99                                   |
| alpha          | 2                                      |
| 标定集大小     | 1024 张（无标签）                      |
| optimizer      | Adam                                   |

**只更新当前 MLP 层的参数，其他层冻结。**

---

### Step 5：保存并替换

蒸馏完成后，当前 MLP 的权重已更新，激活函数已是 ReLU。
继续处理下一个 Block 的 MLP（串行进行）。

---

## 四、代码结构建议

```
mlp_reconstruction/
├── calibrate.py       # 收集标定数据的 O_GELU 和 X
├── aph.py             # 计算 APH 权重 H_bar
├── reconstruct.py     # 蒸馏训练主循环（L_Direct + L_Clamp）
├── replace_gelu.py    # 替换模型中 GELU 为 ReLU 的工具函数
└── run.py             # 主入口：逐层调用上述模块
```

---

## 五、针对 MobileCLIP2 的注意事项

1. **逐层串行**：MobileCLIP2 有 image encoder 和 text encoder，需分别对各自 MLP 层逐层重建。

2. **标定数据**：使用无标签图文对，图文各取 1024 条，前向到对应 encoder 即可，不需要匹配对。

3. **APH 的蒸馏 loss 选择**：对 CLIP 类模型，建议将 MLP 输出的 L2 距离作为蒸馏 loss（逐层局部蒸馏），不需要拉通到最终 contrastive loss，计算更稳定。

4. **text encoder 的 MLP**：text encoder 通常 MLP 层较浅，dying ReLU 风险更低，替换效果会更好。

5. **不量化**：本实施不涉及 `AdaRound`、`QDrop` 等量化操作，只做 GELU→ReLU 替换 + 蒸馏，代码可大幅简化。

---

## 六、验证方法

每完成一个 MLP 层的重建后，建议：

1. 在标定集上比较 `O_GELU` 与 `O_ReLU` 的 cosine similarity，应 > 0.99
2. 在官方 56 张图的验证集上跑 retrieval 指标（Recall@1/5/10），观察精度变化
3. 用 `torch.profiler` 或 `time` 粗测推理速度提升

---

## 七、预期效果（来自论文消融实验）

在仅做 MLP Reconstruction（不量化）的实验中：

| 模型   | 全精度 | MLP Recon 后                     |
| ------ | ------ | -------------------------------- |
| ViT-S  | 81.39% | 80.90%（-0.49%）                 |
| ViT-B  | 84.54% | **84.84%**（+0.30%，超过全精度） |
| DeiT-T | 72.21% | 71.07%（-1.14%）                 |
| DeiT-S | 79.85% | 79.38%（-0.47%）                 |
| Swin-S | 83.23% | 83.12%（-0.11%）                 |

> 多数模型精度损失 < 0.5%，部分甚至提升，同时推理速度提升约 10~20%。

---

_参考论文：APHQ-ViT: Post-Training Quantization with Average Perturbation Hessian Based Reconstruction for Vision Transformers, CVPR 2025_
