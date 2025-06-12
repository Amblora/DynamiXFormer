# DynamiXFormer: A Hybrid Time-Series Forecasting Model with Dynamic Sparse Attention and Adaptive Frequency Denoising

This repository contains the implementation of **DynamiXFormer**, an advanced time-series forecasting model. It integrates several innovative techniques to address key challenges in Long-Sequence Time-Series Forecasting (LSTF), such as high computational complexity, sensitivity to noise, and difficulty in capturing complex temporal dependencies.

## Table of Contents
1. [Introduction](#1-introduction)
2. [Model Architecture & Core Innovations](#2-model-architecture--core-innovations)
    - [2.1. Overall Architecture](#21-overall-architecture)
    - [2.2. Dynamic Sparse Attention](#22-dynamic-sparse-attention)
    - [2.3. Adaptive Frequency Denoise Block (APDC)](#23-adaptive-frequency-denoise-block-apdc)
    - [2.4. Relative Event Embedding](#24-relative-event-embedding)
    - [2.5. Fourier Decomposition & Trend Forecasting](#25-fourier-decomposition--trend-forecasting)
3. [Code Structure Analysis](#3-code-structure-analysis)
4. [How to Use](#4-how-to-use)
5. [Configurable Parameters](#5-configurable-parameters)
6. [Potential Improvements](#6-potential-improvements)

---

## 1. Introduction

DynamiXFormer is a deep learning model designed for time-series forecasting tasks. It is built upon the classic Encoder-Decoder architecture but features deeply customized and innovative core components, making it more efficient and robust when handling complex time-series data.

**Core Features:**

- **Dynamic Sparse Attention**: Replaces the conventional full attention mechanism. It dynamically generates sparse attention patterns based on the intrinsic properties of the data (e.g., volatility, trend changes), significantly reducing computational complexity and memory usage while focusing on the most critical time points.
- **Hybrid Domain Processing**: The model processes signals in both the time and frequency domains simultaneously. It uses Fourier decomposition to separate the series into trend and seasonal components and introduces an Adaptive Frequency Denoise Block (APDC) to denoise and refine features.
- **Data-Driven Embeddings**: An innovative `RelativeEventEmbedding` mechanism captures multi-scale relative changes and inter-event similarities from the raw data, generating a richer feature representation than traditional positional encodings.
- **Hierarchical Forecasting**: Similar to Autoformer and FEDformer, the model predicts the seasonal and trend components separately in the decoder, and their sum forms the final result. This improves both prediction accuracy and interpretability.

---

## 2. Model Architecture & Core Innovations

### 2.1. Overall Architecture

The model follows an Encoder-Decoder structure:

1.  **Input Processing**:
    -   First, the input sequence `x_enc` is decomposed into an initial seasonal component (`seasonal_init`) and a trend component (`trend_init`) using `fourier_decomp`.
    -   `trend_init` serves as the initial value for the cumulative trend in the decoder.
    -   `seasonal_init` is used as the initial input to the decoder to predict future seasonal variations.

2.  **Encoder**:
    -   The input sequence `x_enc` is passed through a `DataEmbedding` layer (which includes the innovative `RelativeEventEmbedding`).
    -   The embedded features are then fed into a stack of `EncoderLayer`s. Each layer uses `DynamicSparseAttention` to capture internal dependencies within the sequence.
    -   Fourier decomposition and the `AdaptiveFreqDenoiseBlock` (APDC) are also integrated within each `EncoderLayer` to denoise and refine features during propagation.

3.  **Decoder**:
    -   The decoder receives the encoder's output `enc_out`, the initial seasonal component `seasonal_init`, and the initial trend component `trend_init`.
    -   Each `DecoderLayer` contains two attention modules:
        -   **Self-Attention**: Uses `DynamicSparseAttention` (with a causal mask) to process the decoder's own input (the seasonal component).
        -   **Cross-Attention**: Uses a standard `FullAttention` to fuse information from the encoder's output.
    -   The decoder also performs Fourier decomposition in each layer, progressively predicting and accumulating the trend component.
    -   The APDC module is also used in the decoder layers to ensure the quality of the output features.

4.  **Output Layer**:
    -   The decoder ultimately outputs the predicted seasonal component and the accumulated trend component.
    -   These two are added together to produce the final prediction `pred_out`.

### 2.2. Dynamic Sparse Attention

This is the core innovation of the model. It determines which keys each query should attend to through a combination of strategies:

-   **Dynamic Local Window**: Based on the local volatility of each time point (calculated via multi-scale differencing), a dynamic look-back window size is determined. More volatile points may get a larger window.
-   **Dynamic Future Window**: A dynamic future-looking window is assigned to each time point based on changes in the sequence's trend.
-   **Keypoint Detection**: Identifies "abrupt changes" or "critical events" in the series by finding local peaks in the rate of change and forces the model to attend to these points.
-   **Stratified Global Sampling**: To prevent the model's view from being too local, this mechanism samples important points from different segments of the sequence based on an importance score (a combination of magnitude, rate of change, and frequency characteristics), ensuring global context is captured.
-   **Random Connection Augmentation**: If the connection density generated by the above strategies is below an adaptive threshold, random connections are added to ensure sufficient information flow.

### 2.3. Adaptive Frequency Denoise Block (APDC)

This module performs fine-grained feature refinement in the frequency domain:

-   **DCT Transform**: Uses the Discrete Cosine Transform (DCT) to convert time-domain features into the frequency domain.
-   **Multi-scale Filtering**: Applies weighting to different frequency bands using multiple learnable weight vectors, achieving multi-scale filtering.
-   **Adaptive Masking**: The module dynamically generates a mask based on frequency energy to automatically identify and suppress noise or unimportant frequency components. It uses different thresholds for (conceptually) regression and classification tasks, allowing the model to preserve different types of high-frequency information as needed.
-   **Convolutional Smoothing**: After filtering, a small convolutional network is used to smooth and enhance the high-frequency parts, further refining the features.

### 2.4. Relative Event Embedding

To replace fixed positional encodings, this module learns temporal relationships from the data itself:

-   **Multi-scale Relative Features**: Calculates feature differences (e.g., distance, energy) across various time spans (scales).
-   **Hybrid Similarity**: Constructs a more robust inter-event similarity metric by combining cosine similarity and Euclidean distance.
-   **Event Attention**: Based on this hybrid similarity, an attention mechanism computes a contextual representation for each event (`PE_event`).
-   The final embedding is a combination of the relative feature encoding and the event context representation, providing the model with exceptionally rich, data-dependent temporal information.

### 2.5. Fourier Decomposition & Trend Forecasting

-   **Decomposition**: Uses the Fast Fourier Transform (FFT) to decompose the series into a low-frequency part (trend) and a high-frequency part (seasonality). The boundary is controlled by the `frequency_threshold` parameter.
-   **Progressive Forecasting**: In the decoder, each layer predicts a `residual_trend`, which is then added to the trend from the previous layers. This progressive refinement makes trend prediction more stable and accurate.

---

## 3. Code Structure Analysis

-   `DynamiXFormer(nn.Module)`: The main model class, integrating the encoder, decoder, and embedding layers.
-   `Encoder(nn.Module)`: The encoder module, composed of a stack of `EncoderLayer`s.
-   `Decoder(nn.Module)`: The decoder module, composed of a stack of `DecoderLayer`s.
-   `EncoderLayer(nn.Module)`: The basic building block of the encoder, containing dynamic sparse attention, a feed-forward network, Fourier decomposition, and APDC.
-   `DecoderLayer(nn.Module)`: The basic building block of the decoder, containing dynamic sparse self-attention, full cross-attention, a feed-forward network, Fourier decomposition, and APDC.
-   `DynamicSparseAttention(nn.Module)`: **Core Innovation**, implementing the dynamic sparse attention mechanism.
-   `AdaptiveFreqDenoiseBlock(nn.Module)`: **Core Innovation**, implementing adaptive frequency-domain denoising.
-   `RelativeEventEmbedding(nn.Module)`: **Core Innovation**, implementing the data-driven event embedding.
-   `DataEmbedding(nn.Module)`: A wrapper class for the total embedding layer, combining `TokenEmbedding`, `PositionalEmbedding`, and `RelativeEventEmbedding`.
-   `fourier_decomp(nn.Module)`: A utility class for performing time-series decomposition.
-   `FullAttention(nn.Module)`: A standard full-attention implementation used for the decoder's cross-attention.
-   `AttentionLayer(nn.Module)`: A generic layer that encapsulates attention computation and projections.

---

## 4. How to Use

Below is an example of how to instantiate and use the `DynamiXFormer` model.

```python
import torch

# 1. Model Configuration
enc_in = 7       # Encoder input feature dimension
dec_in = 7       # Decoder input feature dimension
c_out = 7        # Output feature dimension
seq_len = 96     # Input sequence length
label_len = 48   # Length of the token sequence for the decoder to start with
pred_len = 24    # Prediction sequence length

d_model = 512    # Model dimension
n_heads = 8      # Number of attention heads
e_layers = 2     # Number of encoder layers
d_layers = 1     # Number of decoder layers
d_ff = 2048      # Dimension of the feed-forward network
dropout = 0.1
activation = 'gelu'
series_decomp = 0.1 # Frequency threshold for Fourier decomposition
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
    series_decomp=series_decomp,
    device=device
).to(device)

# 3. Prepare Input Data (dummy data)
batch_size = 32
x_enc = torch.randn(batch_size, seq_len, enc_in).to(device)
x_mark_enc = None # Placeholder for timestamp features, if any

# The decoder input is typically composed of the latter half of the input sequence and zero-padding
dec_input_token = x_enc[:, -label_len:, :]
dec_input_zeros = torch.zeros(batch_size, pred_len, dec_in).to(device)
x_dec = torch.cat([dec_input_token, dec_input_zeros], dim=1)
x_mark_dec = None # Placeholder for decoder timestamp features

# 4. Model Forward Pass
# Masks can be provided during training, but can be None for inference
output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

# 5. Check Output
print(f"Model output shape: {output.shape}")
# Expected output: torch.Size([32, 24, 7]) (batch_size, pred_len, c_out)
```

---

## 5. Configurable Parameters

When instantiating `DynamiXFormer`, you can configure it with the following key parameters:

-   `enc_in`, `dec_in`, `c_out`: Feature dimensions for input and output data.
-   `seq_len`, `label_len`, `pred_len`: Lengths of the input, label, and prediction sequences.
-   `d_model`, `n_heads`, `d_ff`: Core dimensionality parameters of the Transformer.
-   `e_layers`, `d_layers`: The number of layers in the encoder and decoder.
-   `series_decomp`: The frequency threshold for Fourier decomposition. A smaller value results in a smoother trend component.
-   `encoder_apdc`, `decoder_apdc` (bool): Whether to enable the `AdaptiveFreqDenoiseBlock` in the encoder/decoder. Useful for ablation studies.
-   `use_event_embeding_enc`, `use_event_embeding_dec` (bool): Whether to use `RelativeEventEmbedding` in the encoder/decoder embedding layers. Useful for ablation studies.
-   `local_window`, `future_window` (in `DynamicSparseAttention`): The base sizes for the dynamic windows.

---

## 6. Potential Improvements

-   **Performance Optimization**: The `_get_sparse_indices` method contains Python loops. These could potentially be optimized using PyTorch's vectorized operations to accelerate the generation of sparse indices.
-   **Hyperparameter Tuning**: The model's performance is sensitive to hyperparameters like `d_model`, `n_heads`, `series_decomp`, and the `threshold` in `DynamicSparseAttention`. Fine-tuning these for specific datasets is crucial.
-   **Ablation Studies**: The boolean flags like `encoder_apdc` and `use_event_embeding_enc` facilitate ablation studies to validate the effectiveness of each innovative module.
-   **Extensibility**: Explore applying this model to other types of sequential data, such as in Natural Language Processing or audio signal processing.
-   **Alternative Frequency-Domain Tools**: Experiment with other time-frequency analysis tools like the Wavelet Transform instead of DCT, which might offer advantages when dealing with non-stationary signals.

