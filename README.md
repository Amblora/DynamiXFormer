### Table of Contents

1. Introduction
2. Model Architecture & Core Innovations
3. Code Structure Analysis
4. How to Use
5. Configurable Parameters
6. Potential Improvements


### 📖 Introduction

**DynamiXFormer** is a deep learning framework designed specifically for forecasting rock burst risk by analyzing microseismic data. It is built upon the classic Encoder-Decoder architecture but features deeply customized and innovative core components to address the unique challenges in geophysical event prediction.

The model departs from conventional time-based forecasting and introduces a **Disturbance-Driven Paradigm**. Instead of predicting risk at fixed time intervals, it directly maps mining-induced disturbances (i.e., the working face advance distance) to a quantitative risk level, establishing a more physically meaningful and practically relevant forecasting model.

#### Core Features:

- **Disturbance-Driven Paradigm**: Aligns predictions with real-world engineering activities, overcoming the mismatch caused by non-uniform mining operations in fixed-time models.
- **Dynamic Sparse Attention**: Replaces the conventional full attention mechanism. It dynamically generates sparse attention patterns based on the intrinsic properties of microseismic data (e.g., energy release, event clustering), significantly reducing computational complexity while focusing on the most critical precursory signals.
- **Hybrid Domain Processing**: The model processes signals in both the time and frequency domains. It introduces an **Adaptive Frequency Denoise Block (AFDB)** to denoise and refine features from a frequency-domain perspective.
- **Data-Driven Embeddings**: An innovative **`RelativeEventEmbedding`** mechanism captures multi-scale relative changes and inter-event physical similarities from the raw data, generating a richer feature representation than traditional positional encodings.

### 🏛️ Model Architecture & Core Innovations

#### 1. Overall Architecture

The model follows an Encoder-Decoder structure:

- **Input Processing**: The input microseismic sequence `x_enc`, which is benchmarked by advance distance, is passed through a `DataEmbedding` layer. This layer includes the innovative `RelativeEventEmbedding` to generate rich, data-dependent feature representations.
- **Encoder**: The embedded features are fed into a stack of `EncoderLayer`s. Each layer uses `DynamicSparseAttention` to capture complex dependencies within the sequence. An `AdaptiveFreqDenoiseBlock` is also integrated within each `EncoderLayer` to denoise and refine features during propagation.
- **Decoder**: The decoder receives the encoder's output `enc_out` and a context sequence. Each `DecoderLayer` contains two attention modules:
  - **Self-Attention**: Uses `DynamicSparseAttention` (with a causal mask) to process the decoder's own input.
  - **Cross-Attention**: Uses a standard `FullAttention` to fuse information from the encoder's output.
  - The AFDB module is also used in the decoder layers to ensure the quality of the output features.
- **Output Layer**: A final projection layer maps the decoder's output to the predicted risk level for future advance steps.

#### 2. Dynamic Sparse Attention

This is a core innovation of the model. It determines which keys each query should attend to through a combination of data-driven strategies:

- **Dynamic Local Attention**: Based on the local characteristics of each event, a dynamic attention window is learned to capture continuous, short-range causal relationships.
- **Key-point Attention**: Identifies "abrupt changes" or "critical events" in the series by finding local peaks in energy or rate of change, forcing the model to grant these points global attention.
- **Global Attention**: To maintain an awareness of global trends, this mechanism samples a set of global attention points from the entire sequence using a comprehensive importance score.
- **Adaptive Random Connectivity**: Introduces a controlled amount of random connections to prevent the model from over-relying on predefined priors and to enhance generalization.

#### 3. Adaptive Frequency Denoise Block (AFDB)

This module performs fine-grained feature refinement in the frequency domain:

- **DCT Transform**: Uses the Discrete Cosine Transform (DCT) to convert time-domain features into the frequency domain, where energy is often more compacted.
- **Multi-scale Filtering**: Applies learnable weights to different frequency bands, achieving adaptive multi-scale filtering.
- **Adaptive Masking**: The module dynamically generates a mask based on frequency energy to automatically identify and suppress noise while preserving or enhancing salient signal details.

#### 4. Relative Event Embedding

To replace fixed positional encodings, this module learns temporal and physical relationships from the data itself:

- **Multi-scale Relative Features**: Calculates feature differences (e.g., spatial distance, energy magnitude) across various time spans (scales) to capture evolutionary trends.
- **Hybrid Similarity**: Constructs a robust inter-event similarity metric by combining cosine similarity (trend similarity) and Euclidean distance (magnitude similarity).
- **Event Attention**: Based on this hybrid similarity, an attention mechanism computes a contextual representation for each event (`PE_event`).
- The final embedding is a combination of the relative feature encoding and the event context representation, providing the model with exceptionally rich, data-dependent temporal information.

### 🔬 Code Structure Analysis

- `DynamiXFormer(nn.Module)`: The main model class, integrating the encoder, decoder, and embedding layers.
- `Encoder(nn.Module)`: The encoder module, composed of a stack of `EncoderLayer`s.
- `Decoder(nn.Module)`: The decoder module, composed of a stack of `DecoderLayer`s.
- `EncoderLayer(nn.Module)`: The basic building block of the encoder, containing dynamic sparse attention, a feed-forward network, and AFDB.
- `DecoderLayer(nn.Module)`: The basic building block of the decoder, containing dynamic sparse self-attention, full cross-attention, and AFDB.
- **`DynamicSparseAttention(nn.Module)`**: **Core Innovation**, implementing the dynamic sparse attention mechanism.
- **`AdaptiveFreqDenoiseBlock(nn.Module)`**: **Core Innovation**, implementing adaptive frequency-domain denoising.
- **`RelativeEventEmbedding(nn.Module)`**: **Core Innovation**, implementing the data-driven event embedding.
- `DataEmbedding(nn.Module)`: A wrapper class for the total embedding layer, combining `TokenEmbedding`, `PositionalEmbedding`, and `RelativeEventEmbedding`.
- `FullAttention(nn.Module)` & `AttentionLayer(nn.Module)`: Standard attention implementations used for the decoder's cross-attention.

### 🚀 How to Use

Below is an example of how to instantiate and use the `DynamiXFormer` model.

```
import torch
from model import DynamiXFormer # Assuming your model classes are in model.py

# 1. Model Configuration (example values from the paper)
enc_in = 11            # Encoder input feature dimension
dec_in = 11            # Decoder input feature dimension
c_out = 1              # Output feature dimension (e.g., energy level)
seq_len = 96           # Input sequence length (e.g., 96 advance steps)
label_len = 48         # Length of the token sequence for the decoder
pred_len = 24          # Prediction sequence length (e.g., next 24 advance steps)
d_model = 64           # Model dimension
n_heads = 8            # Number of attention heads
e_layers = 2           # Number of encoder layers
d_layers = 1           # Number of decoder layers
d_ff = 128             # Dimension of the feed-forward network
dropout = 0.1
activation = 'gelu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Instantiate the Model
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

# 3. Prepare Input Data (dummy data)
batch_size = 32
x_enc = torch.randn(batch_size, seq_len, enc_in).to(device)
# The decoder input is typically composed of a context part and zero-padding
dec_input_context = x_enc[:, -label_len:, :]
dec_input_placeholder = torch.zeros(batch_size, pred_len, dec_in).to(device)
x_dec = torch.cat([dec_input_context, dec_input_placeholder], dim=1)

# 4. Model Forward Pass
# Time features (x_mark_*) are not used in this specific paradigm but are kept for compatibility
output = model(x_enc, None, x_dec, None)

# 5. Check Output
print(f"Model output shape: {output.shape}")
# Expected output: torch.Size([32, 24, 1]) (batch_size, pred_len, c_out)
```

### ⚙️ Configurable Parameters

When instantiating `DynamiXFormer`, you can configure it with the following key parameters:

- `enc_in`, `dec_in`, `c_out`: Feature dimensions for input and output data.
- `seq_len`, `label_len`, `pred_len`: Lengths of the input, label, and prediction sequences.
- `d_model`, `n_heads`, `d_ff`: Core dimensionality parameters of the Transformer.
- `e_layers`, `d_layers`: The number of layers in the encoder and decoder.
- `encoder_apdc`, `decoder_apdc` (bool): Whether to enable the `AdaptiveFreqDenoiseBlock` in the encoder/decoder. Useful for ablation studies.
- `use_event_embeding_enc`, `use_event_embeding_dec` (bool): Whether to use `RelativeEventEmbedding` in the encoder/decoder. Useful for ablation studies.
- `local_window` (in `DynamicSparseAttention`): The base size for the dynamic local window.

### 💡 Potential Improvements

- **Performance Optimization**: The `_get_sparse_indices` method in `DynamicSparseAttention` contains Python loops. These could potentially be optimized using PyTorch's vectorized operations to accelerate performance.
- **Hyperparameter Tuning**: The model's performance is sensitive to hyperparameters like `d_model`, `n_heads`, and the `threshold` in `DynamicSparseAttention`. Fine-tuning these for specific datasets is crucial.
- **Extensibility**: Apply the framework to other disturbance-driven physical processes, such as seismic monitoring in civil engineering or industrial equipment failure prediction based on operational load.
- **Advanced Relational Modeling**: Explore using Graph Neural Networks (GNNs) to model more complex, non-sequential inter-event relationships, extending the capabilities of the `RelativeEventEmbedding` module.
