# FAST Tokenization Details

## 1.1 Mathematical Framework

The FAST compression pipeline consists of three stages:

### Stage 1: Discrete Cosine Transform (DCT)

Given an action chunk $\mathbf{a} = [a_0, a_1, ..., a_{T-1}] \in \mathbb{R}^{T \times D}$ where $T$ is the temporal horizon and $D$ is the action dimension, DCT is applied independently to each dimension:

$$C_k^{(d)} = \alpha_k \sum_{t=0}^{T-1} a_t^{(d)} \cos\left[\frac{\pi(2t + 1)k}{2T}\right]$$

where $\alpha_k = \sqrt{1/T}$ for $k = 0$ and $\alpha_k = \sqrt{2/T}$ for $k > 0$. This transformation concentrates signal energy into low-frequency coefficients, with typically 90% of energy captured in the first 10-20% of coefficients.

### Stage 2: Adaptive Quantization

The DCT coefficients are quantized using a learnable, non-uniform scheme:

$$\hat{C}_k = \text{Quantize}(C_k/\sigma_k, B_k) \cdot \sigma_k$$

where $\sigma_k$ is the empirical standard deviation of the $k$-th coefficient across the dataset, and $B_k \in \{2, 4, 8, 16\}$ represents adaptive bit allocation. Low-frequency coefficients receive $B_0 = 16$ bits while high frequencies use $B_{T-1} = 2$ bits.

### Stage 3: Byte-Pair Encoding (BPE)

The quantized coefficients are flattened and encoded using BPE with vocabulary size $V = 32,768$.