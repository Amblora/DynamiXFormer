目录

引言

模型架构与核心创新

代码结构分析

如何使用

可配置参数

潜在的改进方向

引用

开源协议

致谢

📖 引言

DynamiXFormer 是一个专为通过分析微震数据来预测岩爆风险而设计的深度学习框架。它基于经典的 Encoder-Decoder 架构，但通过深度定制和创新的核心组件，以应对地球物理事件预测中的独特挑战。

该模型摒弃了传统的基于时间的预测范式，创新性地提出了扰动驱动范式。它不再是在固定的时间间隔上预测风险，而是直接将采矿引起的扰动（即工作面推进距离）映射到量化的风险水平，从而建立了一个更具物理意义和工程实践相关性的预测模型。

核心特性:

扰动驱动范式: 将预测与真实的工程活动对齐，克服了在非匀速采矿作业中，固定时间模型所造成的预测基准失配问题。

动态稀疏注意力: 替代了传统的全注意力机制。它根据微震数据的内在属性（如能量释放、事件聚集性）动态生成稀疏的注意力模式，在显著降低计算复杂度的同时，将计算资源集中在最关键的前兆信号上。

混合域处理: 模型同时在时域和频域中处理信号。它引入了自适应频率去噪模块 (AFDB)，从频域角度对特征进行去噪和提纯。

数据驱动的嵌入: 创新的 RelativeEventEmbedding 机制从原始数据中捕捉多尺度的相对变化和事件间的物理相似性，生成比传统位置编码更丰富的特征表示。

🏛️ 模型架构与核心创新

1. 整体架构

模型遵循 Encoder-Decoder 结构：

输入处理: 以推进距离为基准的微震输入序列 x_enc 被送入 DataEmbedding 层。该层包含了创新的 RelativeEventEmbedding，以生成丰富的、数据驱动的特征表示。

编码器: 嵌入后的特征被送入一个由 EncoderLayer 堆叠而成的编码器。每个层级都使用 DynamicSparseAttention 来捕捉序列内部的复杂依赖关系。同时，AdaptiveFreqDenoiseBlock 也被集成在每个 EncoderLayer 中，用于在信息传播过程中进行去噪和特征提纯。

解码器: 解码器接收编码器的输出 enc_out 和一个上下文序列。每个 DecoderLayer 包含两个注意力模块：

自注意力: 使用 DynamicSparseAttention (带有因果掩码) 来处理解码器自身的输入。

交叉注意力: 使用标准的 FullAttention 来融合来自编码器输出的信息。

AFDB 模块同样被用于解码器层，以保证输出特征的质量。

输出层: 最终的投影层将解码器的输出映射为对未来推进步骤的风险等级预测。

2. 动态稀疏注意机制

这是模型的核心创新。它通过多种数据驱动策略的组合来决定每个查询（query）应该关注哪些键（key）：

动态局部注意力: 根据每个事件的局部特征，学习一个动态的注意力窗口，以捕捉连续的、短程的因果关系。

关键点注意力: 通过寻找能量或变化率的局部峰值来识别序列中的“突变”或“关键事件”，并强制模型给予这些点全局的关注。

全局注意力: 为了保持对全局趋势的感知，该机制使用一个综合重要性分数，从整个序列中采样一组全局注意力点。

自适应随机连接: 引入了受控的随机连接，以防止模型过度依赖预设的先验知识，并增强泛化能力。

3. 自适应频率去噪模块 (AFDB)

该模块在频域中执行精细的特征提纯：

DCT变换: 使用离散余弦变换（DCT）将时域特征转换到频域，频域中的能量通常更集中。

多尺度滤波: 使用可学习的权重对不同的频带进行加权，实现自适应的多尺度滤波。

自适应掩码: 模块根据频率能量动态生成掩码，以自动识别和抑制噪声，同时保留或增强显著的信号细节。

4. 相对事件嵌入

为了取代固定的位置编码，该模块从数据本身学习时间和物理关系：

多尺度相对特征: 计算不同时间跨度（尺度）下的特征差异（如空间距离、能量大小），以捕捉演化趋势。

混合相似度: 通过结合余弦相似度（趋势相似性）和欧氏距离（数值相似性），构建了一个更稳健的事件间相似性度量。

事件注意力: 基于这种混合相似度，一个注意力机制为每个事件计算上下文表示（PE_event）。

最终的嵌入是相对特征编码和事件上下文表示的组合，为模型提供了异常丰富的、依赖于数据的时序信息。

🔬 代码结构分析

DynamiXFormer(nn.Module): 主模型类，集成了编码器、解码器和嵌入层。

Encoder(nn.Module): 编码器模块，由一系列 EncoderLayer 堆叠而成。

Decoder(nn.Module): 解码器模块，由一系列 DecoderLayer 堆叠而成。

EncoderLayer(nn.Module): 编码器的基本构建块，包含动态稀疏注意力、前馈网络和AFDB。

DecoderLayer(nn.Module): 解码器的基本构建块，包含动态稀疏自注意力、全交叉注意力和AFDB。

DynamicSparseAttention(nn.Module): 核心创新，实现了动态稀疏注意机制。

AdaptiveFreqDenoiseBlock(nn.Module): 核心创新，实现了自适应频域去噪。

RelativeEventEmbedding(nn.Module): 核心创新，实现了数据驱动的事件嵌入。

DataEmbedding(nn.Module): 嵌入层的包装类，整合了 TokenEmbedding, PositionalEmbedding, 和 RelativeEventEmbedding。

FullAttention(nn.Module) & AttentionLayer(nn.Module): 标准的全注意力实现，用于解码器的交叉注意力。

🚀 如何使用

以下是一个如何实例化并使用 DynamiXFormer 模型的示例。

import torch
from model import DynamiXFormer # 假设您的模型类位于 model.py 文件中

# 1. 模型配置 (参考论文中的示例值)
enc_in = 11            # 编码器输入特征维度
dec_in = 11            # 解码器输入特征维度
c_out = 1              # 输出特征维度 (例如：能量水平)
seq_len = 96           # 输入序列长度 (例如：96个推进单位)
label_len = 48         # 解码器用于上下文的序列长度
pred_len = 24          # 预测序列长度 (例如：未来24个推进单位)
d_model = 64           # 模型维度
n_heads = 8            # 注意力头数
e_layers = 2           # 编码器层数
d_layers = 1           # 解码器层数
d_ff = 128             # 前馈网络维度
dropout = 0.1
activation = 'gelu'
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
    device=device
).to(device)

# 3. 准备输入数据 (模拟数据)
batch_size = 32
x_enc = torch.randn(batch_size, seq_len, enc_in).to(device)
# 解码器的输入通常由输入序列的后半部分和零填充组成
dec_input_context = x_enc[:, -label_len:, :]
dec_input_placeholder = torch.zeros(batch_size, pred_len, dec_in).to(device)
x_dec = torch.cat([dec_input_context, dec_input_placeholder], dim=1)

# 4. 模型前向传播
# 时间特征 (x_mark_*) 在此特定范式中未使用，但保留以兼容
output = model(x_enc, None, x_dec, None)

# 5. 检查输出
print(f"模型输出形状: {output.shape}")
# 预期输出: torch.Size([32, 24, 1]) (batch_size, pred_len, c_out)


⚙️ 可配置参数

在实例化 DynamiXFormer 时，您可以通过以下关键参数进行配置：

enc_in, dec_in, c_out: 输入和输出数据的特征维度。

seq_len, label_len, pred_len: 输入、标签和预测序列的长度。

d_model, n_heads, d_ff: Transformer 的核心维度参数。

e_layers, d_layers: 编码器和解码器的层数。

encoder_apdc, decoder_apdc (布尔值): 是否在编码器/解码器中启用 AdaptiveFreqDenoiseBlock。可用于消融实验。

use_event_embeding_enc, use_event_embeding_dec (布尔值): 是否在编码器/解码器中使用 RelativeEventEmbedding。可用于消融实验。

local_window (在 DynamicSparseAttention 中): 动态局部窗口的基础尺寸。

💡 潜在的改进方向

性能优化: DynamicSparseAttention 中的 _get_sparse_indices 方法包含Python循环。这些循环有潜力通过PyTorch的向量化操作进行优化，以提升性能。

超参数调优: 模型的性能对 d_model, n_heads 以及 DynamicSparseAttention 中的 threshold 等超参数敏感。针对特定数据集进行微调至关重要。

可扩展性: 将此框架应用于其他扰动驱动的物理过程，例如土木工程中的地震监测，或基于运行负载的工业设备故障预测。

更高级的关系建模: 探索使用图神经网络（GNNs）来建模更复杂的、非序列化的事件间关系，以扩展 RelativeEventEmbedding 模块的能力。
