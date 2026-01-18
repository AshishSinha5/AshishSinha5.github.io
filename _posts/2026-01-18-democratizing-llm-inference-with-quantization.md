---
layout: post
title: "Democratizing LLM Inference with Quantization Techniques"
date: 2026-01-18 10:00:00 +0530
category: blog
headerImage: false
tag:
    - LLM
    - Quantization
    - Machine Learning
    - Deep Learning
    - Transformers
---

{%- include mathjax.html -%}
# Democratizing LLM Inference with Quantization Techniques

## Introduction

Large Language Models (LLMs) have become a powerful interface for reasoning, search, and content generation. However, the ability to *run* these models remains restricted by hardware. A large fraction of value created by LLMs is currently gated behind access to high-memory GPUs.

This is not only a deployment problem. It directly impacts democratization:

- Individuals and small teams cannot afford multi-GPU setups
- Running models locally on consumer GPUs, laptops, or edge devices becomes infeasible
- Even in cloud settings, inference cost scales with memory bandwidth and model size

Quantization addresses this at a fundamental level. It compresses model weights and activations into lower precision representations while preserving most of the model’s capability. In many cases, quantization is the only reason models can be served cheaply at scale or run locally.

This blog explains:
1. Why quantization is critical for democratizing LLMs  
2. The core quantization formulation  
3. The main techniques used in practice  
4. How to implement them (with examples)

---

## Why Quantization is Required for Democratization

### 1) Memory is the hard bottleneck

A model with $N$ parameters stored in FP16 requires approximately:

$$
\text{Memory} \approx 2N \text{ bytes}
$$

Examples:

| Model Size | Precision | Approx. Weight Memory (Weights Only) |
|-----------:|:---------:|--------------------------------------:|
| 7B         | FP16      | $\approx 14$ GB                       |
| 13B        | FP16      | $\approx 26$ GB                       |
| 70B        | FP16      | $\approx 140$ GB                      |

This does not include:
- KV cache (grows with sequence length)
- optimizer states (if training)
- temporary activations

Even inference becomes impossible on common GPUs without quantization.

---

### 2) Inference cost is dominated by memory bandwidth

Most transformer inference kernels are memory-bound:
- fetching weights from HBM/DRAM dominates time
- lower precision reduces bandwidth and improves throughput

Quantization often improves:
- latency
- throughput
- batch scalability

---

### 3) Quantization makes on-device and edge inference possible

Running LLMs on:
- laptops (16GB RAM)
- consumer GPUs (8–12GB VRAM)
- mobile/edge accelerators

is only possible via weight compression and low-bit math.

---

## Quantization Basics

Quantization maps a floating tensor $x$ to a lower precision representation $q$.

### Uniform affine quantization

Given scale $s$ and zero-point $z$:

$$
q = \text{clip}\left(\left\lfloor \frac{x}{s} \right\rceil + z, q_{\min}, q_{\max}\right)
$$

Dequantization:

$$
\hat{x} = s (q - z)
$$

Common choices:
- INT8: $q \in [-128, 127]$ or $[0,255]$
- INT4: $q \in [0, 15]$


Quantization error:

$$
\epsilon = x - \hat{x}
$$

The goal is to choose quantization parameters so that $\epsilon$ minimally harms downstream model quality.

---

## Key Design Axes of Quantization

### 1) Weight vs Activation quantization
- **Weight-only quantization**: easiest and most common for LLM inference
- **Weight + activation quantization**: harder, but yields better speedups on accelerators

### 2) Per-tensor vs Per-channel quantization
- **Per-tensor**: one scale for the whole tensor
- **Per-channel**: one scale per output channel (/row) (better accuracy)

### 3) Symmetric vs Asymmetric
- Symmetric uses $z=0$ and ranges like $[-127,127]$
- Asymmetric uses non-zero $z$ and covers non-centered distributions better

---

## 1) Post-Training Quantization (PTQ)

PTQ quantizes a model *after training* without additional gradient updates.

It is the most accessible technique because it requires:
- a trained model
- a small calibration dataset (optional for weight-only PTQ)

### Simple weight-only INT8 PTQ example (PyTorch)

```python
import torch

def symmetric_int8_quantize(w: torch.Tensor):
    # scale based on max absolute value
    max_val = w.abs().max()
    s = max_val / 127.0 + 1e-12
    q = torch.clamp((w / s).round(), -127, 127).to(torch.int8)
    w_hat = q.float() * s
    return q, s, w_hat

W = torch.randn(1024, 1024)
q, s, W_hat = symmetric_int8_quantize(W)

err = (W - W_hat).abs().mean().item()
print("Mean absolute error:", err)
```
### How weights go from higher → lower precision (Quantization summary)

Given high-precision weights `w` (FP16/FP32), quantization stores them in a lower-precision format like INT8 using **scale + rounding**.

#### Step 1: Compute the scale
Find the maximum absolute value in the tensor:

- `max_val = max(|w|)`

For symmetric INT8 quantization, values must fit in `[-127, 127]`, so:

- `s = max_val / 127`

This means **one INT8 step ≈ `s` in float space**.

---

#### Step 2: Convert float weights to INT8 codes
Normalize and round:

- `q = round(w / s)`

Then clamp into valid INT8 range:

- `q = clamp(q, -127, 127)`

Now `q` is stored as **INT8**, reducing memory.

---

#### Step 3: Dequantize back to float (approximation)
To use the quantized weights in computation:

- `w_hat = q * s`

Here `w_hat` is the float approximation of the original weights.

---

#### Key intuition
Quantization forces weights to lie on a discrete grid:

- `w_hat ∈ { -127s, ..., -1s, 0, 1s, ..., 127s }`

So you gain **memory + speed**, but introduce **quantization error**:

- `error = w - w_hat`

### Interpretation

- This compresses weights by 2× compared to FP16 (INT8 vs FP16)
- Accuracy depends heavily on:
  - per-channel scaling
  - outlier handling

---

## 2) Per-Channel Quantization (Better PTQ)

In transformer MLP and attention projections, each output channel often has different statistics.

Per-channel scaling significantly reduces quantization error.

```python
def per_channel_int8_quantize(W: torch.Tensor, dim=0):
    # quantize per output channel
    # W shape: (out_features, in_features)
    max_val = W.abs().amax(dim=dim, keepdim=True)
    s = max_val / 127.0 + 1e-12
    q = torch.clamp((W / s).round(), -127, 127).to(torch.int8)
    W_hat = q.float() * s
    return q, s, W_hat
```
This is commonly used by practical quantizers because it preserves directional information better.

---

## 3) Quantization-Aware Training (QAT)

PTQ is limited because it does not allow the model to adapt to quantization noise.

QAT simulates quantization during training:

$$
\hat{W} = \text{Quantize}(W)
$$

and trains the model to perform well with quantized weights.
The key trick is the **Straight-Through Estimator (STE)**:

$$\frac{\partial \hat{W}}{\partial W} \approx 1$$

This allows gradients to flow through the quantization operation.
### Simple QAT example (PyTorch)

```python
class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))  
        self.quant_func = symmetric_int8_quantize   
    def forward(self, x):
        # Simulate quantization
        q, s, w_hat = self.quant_func(self.weight)
        return F.linear(x, w_hat)
```

We have a simple linear layer that quantizes weights on-the-fly during the forward pass. In practice, QAT yields higher accuracy but is more expensive and less commonly used for LLMs due to training cost.

---

## 4) GPTQ (Second-Order Weight Only Quantization)

LLM quantization in practice is dominated by weight-only 4-bit methods.

GPTQ is a widely used approach that quantizes weights sequentially while minimizing output error using second-order information (approximated Hessian).

### Intuition
For a layer: 

$$
y = W x
$$

We want to quantize $W$ into $\hat{W}$ such that:
$$
||Wx - \hat{W}x||
$$
is minimized on calibration data.

GPTQ performs:

- blockwise quantization
- error compensation during sequential updates
- focuses on preserving layer outputs

Practical summary (why it works)

- handles outliers better than naive PTQ
- preserves accuracy at 4-bit
- widely used with LLaMA-class models

---

## 5) AWQ (Activation-Aware Weight Quantization)
A problem with 4-bit weight quantization is that a few channels dominate activation magnitude.

AWQ identifies these channels and rescales weights to reduce quantization damage.

### Key idea
If a channel has large activation magnitude, quantization error on corresponding weights gets amplified.

AWQ uses calibration data to find important channels and rescales:

$$
W_i' = W . \alpha_i
$$
so that quantization error is distributed more evenly.

AWQ often improves 4-bit performance without training and it is perticularly effective for attention/MLP layers with activation outliers.

---

## 6) SmoothQuant (Weight-Activation Balancing)
Activation quantization is harder than weight quantization due to outliers.

SmoothQuant addresses this by moving scale between activation and weight:

$$
XW = (X S^{-1})(S W)
$$

where $S$ is a diagonal scaling matrix.

The model is algebraically unchanged, but now activations are better conditioned for quantization.

Even though the function is unchanged, the **internal magnitudes** change:

- Activations become: $X' = XS^{-1}$
- Weights become: $W' = SW$

This redistributes scale between activations and weights.

---

### Why this helps quantization

Quantization error depends heavily on **dynamic range** and **outliers**.

- Activations in LLMs often contain large spikes (outliers)
- INT8 activation quantization must pick a scale that covers those outliers
- This compresses most normal values into few bins → large rounding error

SmoothQuant reduces activation outliers by scaling them down:

- activations become more “quantization friendly”
- weights become slightly larger, but weight quantization is typically easier
  (due to per-channel scaling and more stable distributions)

---

## 7) 4-bit Quantization with NF4 (QLoRA-style)

QLoRA popularized fine-tuning in 4-bit by using:

- 4-bit NormalFloat (NF4) quantization for weights
- LoRA adapters for training

This enables fine-tuning large models on limited GPUs.

The core pattern:

- store base weights in 4-bit
- keep LoRA trainable weights in FP16/BF16

This is one of the most important democratization techniques:

- enables experimentation on consumer GPUs
- keeps quality close to full fine-tuning

---

## Practical Guidelines

1. If cheap inference is the goal use weight only-only 4 bit (GPTQ/AWQ/NF4) and keep KV cache in FP16.
2. If the goal is to fine tune with limited GPU memory, use QLoRA (4-bit NF4 + LoRA adapters).
3. If the use-case is accelerator friendly deployment, consider SmoothQuant + INT8 activation quantization.

---

## Conclusion
If LLMs are to become truly accessible, quantization is not optional. It is the primary mechanism that enables:
- local execution 
- low-cost serving
- consumer hardware compatibility
- broader experimentation outside large organizations

Modern quantization methods (GPTQ, AWQ, SmoothQuant, NF4) represent a shift from naive rounding to structure-aware compression strategies that preserve model behavior under extreme precision constraints.

As model sizes continue to grow, quantization is becoming a foundational requirement for democratizing model usage and development.

---

## References
1. Frantar et al., GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers, NeurIPS 2023
2. Dettmers et al., QLoRA: Efficient Finetuning of Quantized LLMs, NeurIPS 2023
3. Lin et al., AWQ: Activation-Aware Weight Quantization for Large Language Models
4. Xiao et al., SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models, ICML 2023
5. Jacob et al., Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference, CVPR 2018
