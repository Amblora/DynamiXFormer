# DynamiXFormer: 动态稀疏注意力与自适应频域降噪的混合时序预测模型

本项目实现了一个名为 **DynamiXFormer** 的先进时间序列预测模型。该模型融合了多种创新技术，旨在解决长序列时间序列预测（LSTF）中的关键挑战，如计算复杂度高、对噪声敏感以及难以捕捉复杂的时序依赖关系等问题。

## 目录
1. [项目简介](#1-项目简介)
2. [模型架构与核心创新](#2-模型架构与核心创新)
    - [2.1. 整体架构](#21-整体架构)
    - [2.2. 动态稀疏注意力 (Dynamic Sparse Attention)](#22-动态稀疏注意力-dynamic-sparse-attention)
    - [2.3. 自适应频域降噪 (Adaptive Frequency Denoise Block - APDC)](#23-自适应频域降噪-adaptive-frequency-denoise-block---apdc)
    - [2.4. 相对事件嵌入 (Relative Event Embedding)](#24-相对事件嵌入-relative-event-embedding)
    - [2.5. 傅里叶分解与趋势预测](#25-傅里叶分解与趋势预测)
3. [代码结构解析](#3-代码结构解析)
4. [使用方法](#4-使用方法)
5. [可配置参数](#5-可配置参数)
6. [潜在改进方向](#6-潜在改进方向)

---

## 1. 项目简介

DynamiXFormer 是一个为时间序列预测任务设计的深度学习模型。它基于经典的 Encoder-Decoder 架构，但对其核心组件进行了深度定制和创新，使其在处理复杂时间序列数据时更加高效和鲁棒。

**核心特性:**

- **动态稀疏注意力**: 替代了传统 Transformer 的全注意力，根据数据本身的特性（如波动性、趋势变化）动态生成稀疏的注意力矩阵，显著降低了计算复杂度和内存消耗，同时聚焦于最重要的时间点。
- **混合域处理**: 模型同时在时域和频域上对信号进行处理。通过傅里叶分解将序列分解为趋势项和季节项，并引入自适应频域降噪模块（APDC）对特征进行去噪和提纯。
- **数据驱动的嵌入**: 创新的相对事件嵌入机制，能够从原始数据中捕捉多尺度的相对变化和事件间的相似性，生成比传统位置编码更丰富的特征表示。
- **分层预测**: 类似于 Autoformer 和 FEDformer，模型在解码器中对季节项和趋势项分别进行预测，最后相加得到最终结果，提高了预测的准确性和可解释性。

---

## 2. 模型架构与核心创新

### 2.1. 整体架构

模型遵循 Encoder-Decoder 结构：

1.  **输入处理**:
    -   首先，使用傅里叶分解 (`fourier_decomp`) 将输入序列 `x_enc` 分解为初始的季节项 (`seasonal_init`) 和趋势项 (`trend_init`)。
    -   `trend_init` 将作为解码器中累积趋势的初始值。
    -   `seasonal_init` 将作为解码器的初始输入，用于预测未来的季节性变化。

2.  **编码器 (Encoder)**:
    -   输入序列 `x_enc` 经过 `DataEmbedding` 层（包含创新的 `RelativeEventEmbedding`）进行嵌入。
    -   嵌入后的特征被送入多层 `EncoderLayer`。每一层都使用 `DynamicSparseAttention` 来捕捉序列内部的依赖关系。
    -   在每个 `EncoderLayer` 中，还集成了傅里叶分解和 `AdaptiveFreqDenoiseBlock` (APDC)，以在特征传递过程中进行去噪和细化。

3.  **解码器 (Decoder)**:
    -   解码器接收编码器的输出 `enc_out`、初始季节项 `seasonal_init` 和初始趋势项 `trend_init`。
    -   每一层 `DecoderLayer` 包含两个注意力模块：
        -   **自注意力**: 使用 `DynamicSparseAttention`（带掩码）处理解码器自身的输入（季节项）。
        -   **交叉注意力**: 使用标准的 `FullAttention` 来融合编码器的输出信息。
    -   解码器同样在每一层中进行傅里叶分解，并逐步预测和累加趋势项。
    -   APDC模块也被用于解码器层，以保证输出特征的质量。

4.  **输出层**:
    -   解码器最终输出预测的季节项和累积的趋势项。
    -   两者相加得到最终的预测结果 `pred_out`。

### 2.2. 动态稀疏注意力 (Dynamic Sparse Attention)

这是模型的核心，它通过多种策略组合来确定每个查询（Query）应该关注哪些键（Key）：

-   **动态局部窗口**: 根据每个时间点的局部波动性（通过多尺度差分计算），动态确定一个回顾窗口大小。波动剧烈的地方，窗口可能更大。
-   **动态未来窗口**: 根据序列的趋势变化，为每个时间点分配一个动态的未来探查窗口。
-   **关键点检测**: 通过寻找变化率的局部峰值来识别序列中的“突变点”或“关键事件”，并强制模型关注这些点。
-   **分层全局采样**: 为了避免模型视野过于局部，该机制从序列的不同分段中，依据重要性得分（综合考虑幅值、变化率和频率特性）进行采样，确保捕获全局上下文。
-   **随机连接补充**: 在上述策略生成的稀疏连接基础上，如果连接密度低于某个自适应阈值，会随机增加一些连接，以保证信息流的充分性。

### 2.3. 自适应频域降噪 (Adaptive Frequency Denoise Block - APDC)

此模块在频域对特征进行精细化处理：

-   **DCT变换**: 使用离散余弦变换（DCT）将时域特征转换到频域。
-   **多尺度滤波**: 通过多个可学习的权重向量，对不同频段的信号进行加权，实现多尺度滤波。
-   **自适应掩码**: 模块能根据频域能量动态生成一个掩码，自动识别并抑制噪声或不重要的频率分量。它为回归和分类任务（概念上的）设置了不同的阈值，允许模型根据任务需求保留不同类型的高频信息。
-   **卷积平滑**: 在滤波后，使用小型卷积网络对高频部分进行平滑和增强，进一步提纯特征。

### 2.4. 相对事件嵌入 (Relative Event Embedding)

为了取代固定的位置编码，该模块从数据自身学习时序关系：

-   **多尺度相对特征**: 计算不同时间跨度（scale）下的特征相对差异（如距离、能量）。
-   **混合相似度**: 结合余弦相似度和欧氏距离，构建一个更鲁棒的事件间相似度度量。
-   **事件注意力**: 基于混合相似度，通过一个注意力机制计算每个事件的上下文表示（`PE_event`）。
-   最终的嵌入是相对特征编码和事件上下文表示的结合，为模型提供了极为丰富的、与数据内容相关的时序信息。

### 2.5. 傅里叶分解与趋势预测

-   **分解**: 使用快速傅里叶变换（FFT）将序列分解为低频部分（趋势项）和高频部分（季节项）。分解的边界由 `frequency_threshold` 参数控制。
-   **渐进式预测**: 在解码器中，每一层都会预测出一个残差趋势（`residual_trend`），并将其累加到之前的趋势项上。这种渐进式修正的方式使得趋势预测更加稳定和准确。

---

## 3. 代码结构解析

-   `DynamiXFormer(nn.Module)`: 主模型，整合了编码器、解码器和嵌入层。
-   `Encoder(nn.Module)`: 编码器模块，由多个 `EncoderLayer` 堆叠而成。
-   `Decoder(nn.Module)`: 解码器模块，由多个 `DecoderLayer` 堆叠而成。
-   `EncoderLayer(nn.Module)`: 编码器基本单元，包含动态稀疏注意力、前馈网络、傅里叶分解和APDC。
-   `DecoderLayer(nn.Module)`: 解码器基本单元，包含动态稀疏自注意力、全交叉注意力、前馈网络、傅里叶分解和APDC。
-   `DynamicSparseAttention(nn.Module)`: **核心创新**，实现了动态稀疏注意力机制。
-   `AdaptiveFreqDenoiseBlock(nn.Module)`: **核心创新**，实现了自适应频域降噪。
-   `RelativeEventEmbedding(nn.Module)`: **核心创新**，实现了数据驱动的事件嵌入。
-   `DataEmbedding(nn.Module)`: 封装了 `TokenEmbedding`, `PositionalEmbedding` 和 `RelativeEventEmbedding` 的总嵌入层。
-   `fourier_decomp(nn.Module)`: 实现时序分解的工具类。
-   `FullAttention(nn.Module)`: 标准的全注意力实现，用于解码器的交叉注意力。
-   `AttentionLayer(nn.Module)`: 对注意力计算和投影进行封装的通用层。

---

## 4. 使用方法

以下是一个如何实例化并使用 `DynamiXFormer` 模型的示例。

```python
import torch

# 1. 模型参数配置
enc_in = 7  # 编码器输入特征维度
dec_in = 7  # 解码器输入特征维度
c_out = 7   # 输出特征维度
seq_len = 96  # 输入序列长度
label_len = 48  # 解码器中用于引导的序列长度 (token length)
pred_len = 24  # 预测序列长度

d_model = 512  # 模型维度
n_heads = 8    # 注意力头数
e_layers = 2   # 编码器层数
d_layers = 1   # 解码器层数
d_ff = 2048  # 前馈网络维度
dropout = 0.1
activation = 'gelu'
series_decomp = 0.1 # 傅里叶分解的频率阈值
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. 实例化模型
model = DynamiXFormer(
    enc_in=enc_in,
    dec_in=dec_in,
    c_out=c_out,
    seq_len=seq_len,
    label_len=label_len,
    pred_len=pred_len,
    d_model=d_model,
    n_heads=n_heads,
    e_layers=e_layers,
    d_layers=d_layers,
    d_ff=d_ff,
    dropout=dropout,
    activation=activation,
    series_decomp=series_decomp,
    device=device
).to(device)

# 3. 准备输入数据 (dummy data)
batch_size = 32
x_enc = torch.randn(batch_size, seq_len, enc_in).to(device)
x_mark_enc = None # 如果有时间戳特征，可以在这里提供

# 解码器输入通常由输入序列的后半部分和零填充组成
dec_input_token = x_enc[:, -label_len:, :]
dec_input_zeros = torch.zeros(batch_size, pred_len, dec_in).to(device)
x_dec = torch.cat([dec_input_token, dec_input_zeros], dim=1)
x_mark_dec = None # 同样，用于解码器的时间戳特征

# 4. 模型前向传播
# 在训练时，通常会提供掩码，但对于推理，可以为None
output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

# 5. 检查输出
print(f"模型输出形状: {output.shape}")
# 预期输出: torch.Size([32, 24, 7]) (batch_size, pred_len, c_out)

```

---

## 5. 可配置参数

在实例化 `DynamiXFormer` 时，可以通过以下关键参数进行配置：

-   `enc_in`, `dec_in`, `c_out`: 输入、输出数据的特征维度。
-   `seq_len`, `label_len`, `pred_len`: 输入、标签和预测序列的长度。
-   `d_model`, `n_heads`, `d_ff`: Transformer 的核心维度参数。
-   `e_layers`, `d_layers`: 编码器和解码器的层数。
-   `series_decomp`: 傅里叶分解的频率阈值。值越小，趋势项包含的频率成分越低（趋势更平滑）。
-   `encoder_apdc`, `decoder_apdc` (bool): 是否在编码器/解码器中启用 `AdaptiveFreqDenoiseBlock`。可用于消融实验。
-   `use_event_embeding_enc`, `use_event_embeding_dec` (bool): 是否在编码器/解码器嵌入层中使用 `RelativeEventEmbedding`。可用于消融实验。
-   `local_window`, `future_window` (在 `DynamicSparseAttention` 中): 控制动态窗口的基础大小。

---

## 6. 潜在改进方向

-   **性能优化**: `_get_sparse_indices` 方法中包含Python循环，可以尝试通过`torch`的向量化操作进行优化，以加速稀疏索引的生成。
-   **超参数调优**: 模型的性能对 `d_model`, `n_heads`, `series_decomp` 以及 `DynamicSparseAttention` 中的 `threshold` 等超参数敏感，需要针对具体数据集进行细致调优。
-   **消融研究**: 利用 `encoder_apdc`, `use_event_embeding_enc` 等布尔标志，可以方便地进行消融研究，以验证每个创新模块的有效性。
-   **可扩展性**: 可以探索将此模型应用于其他序列数据类型，如自然语言处理或音频信号。
-   **频域分析工具**: 尝试使用小波变换（Wavelet Transform）等其他时频分析工具替代DCT，可能会在处理非平稳信号时带来优势。

