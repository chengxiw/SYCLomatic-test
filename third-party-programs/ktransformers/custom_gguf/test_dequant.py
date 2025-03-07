import numpy as np

# add path
# import sys
# sys.path.append("../..")

# from ktransformers.util.custom_gguf import dequantize_q4_k_gpu
import torch
import dequantize_extension
torch.set_default_dtype(torch.float32)
import time
from transformers import (
    AutoConfig,
)
from enum import IntEnum
from typing import Sequence
import ctypes

class GGMLQuantizationType(IntEnum):
    F32     = 0
    F16     = 1
    Q4_0    = 2
    Q4_1    = 3
    Q5_0    = 6
    Q5_1    = 7
    Q8_0    = 8
    Q8_1    = 9
    Q2_K    = 10
    Q3_K    = 11
    Q4_K    = 12
    Q5_K    = 13
    Q6_K    = 14
    Q8_K    = 15
    IQ2_XXS = 16
    IQ2_XS  = 17
    IQ3_XXS = 18
    IQ1_S   = 19
    IQ4_NL  = 20
    IQ3_S   = 21
    IQ2_S   = 22
    IQ4_XS  = 23
    I8      = 24
    I16     = 25
    I32     = 26
    I64     = 27
    F64     = 28
    IQ1_M   = 29
    BF16    = 30

QK_K = 256
GGML_QUANT_SIZES: dict[GGMLQuantizationType, tuple[int, int]] = {
    GGMLQuantizationType.F32:     (1, 4),
    GGMLQuantizationType.F16:     (1, 2),
    GGMLQuantizationType.Q4_0:    (32, 2 + 16),
    GGMLQuantizationType.Q4_1:    (32, 2 + 2 + 16),
    GGMLQuantizationType.Q5_0:    (32, 2 + 4 + 16),
    GGMLQuantizationType.Q5_1:    (32, 2 + 2 + 4 + 16),
    GGMLQuantizationType.Q8_0:    (32, 2 + 32),
    GGMLQuantizationType.Q8_1:    (32, 4 + 4 + 32),
    GGMLQuantizationType.Q2_K:    (256, 2 + 2 + QK_K // 16 + QK_K // 4),
    GGMLQuantizationType.Q3_K:    (256, 2 + QK_K // 4 + QK_K // 8 + 12),
    GGMLQuantizationType.Q4_K:    (256, 2 + 2 + QK_K // 2 + 12),
    GGMLQuantizationType.Q5_K:    (256, 2 + 2 + QK_K // 2 + QK_K // 8 + 12),
    GGMLQuantizationType.Q6_K:    (256, 2 + QK_K // 2 + QK_K // 4 + QK_K // 16),
    GGMLQuantizationType.Q8_K:    (256, 4 + QK_K + QK_K // 8),
    GGMLQuantizationType.IQ2_XXS: (256, 2 + QK_K // 4),
    GGMLQuantizationType.IQ2_XS:  (256, 2 + QK_K // 4 + QK_K // 32),
    GGMLQuantizationType.IQ3_XXS: (256, 2 + QK_K // 4 + QK_K // 8),
    GGMLQuantizationType.IQ1_S:   (256, 2 + QK_K // 8 + QK_K // 16),
    GGMLQuantizationType.IQ4_NL:  (32, 2 + 16),
    GGMLQuantizationType.IQ3_S:   (256, 2 + QK_K // 4 + QK_K // 8 + QK_K // 32 + 4),
    GGMLQuantizationType.IQ2_S:   (256, 2 + QK_K // 4 + QK_K // 16),
    GGMLQuantizationType.IQ4_XS:  (256, 2 + 2 + QK_K // 2 + QK_K // 64),
    GGMLQuantizationType.I8:      (1, 1),
    GGMLQuantizationType.I16:     (1, 2),
    GGMLQuantizationType.I32:     (1, 4),
    GGMLQuantizationType.I64:     (1, 8),
    GGMLQuantizationType.F64:     (1, 8),
    GGMLQuantizationType.IQ1_M:   (256, QK_K // 8 + QK_K // 16  + QK_K // 32),
    GGMLQuantizationType.BF16:    (1, 2),
}

# copied from llama.cpp/gguf-py/gguf/quants.py to avoid dependence of gguf
def quant_shape_to_byte_shape(shape: Sequence[int], quant_type: GGMLQuantizationType):
    block_size, type_size = GGML_QUANT_SIZES[quant_type]
    if shape[-1] % block_size != 0:
        raise ValueError(f"Quantized tensor row size ({shape[-1]}) is not a multiple of {quant_type.name} block size ({block_size})")
    return (*shape[:-1], shape[-1] // block_size * type_size)

GGML_TYPES = {
    "F32": 0,
    "F16": 1,
    "Q4_0": 2,
    "Q5_0": 6,
    "Q8_0": 8,
    "Q2_K": 10,
    "Q3_K": 11,
    "Q4_K": 12,
    "Q5_K": 13,
    "Q6_K": 14,
    "IQ4_XS": 23,
    "BF16": 30,
}

GGML_NAMES = {ggml_type: name for name, ggml_type in GGML_TYPES.items()}

GGML_BLOCK_SIZES = {
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "Q4_0": 2 + 16,
    "Q5_0": 2 + 4 + 16,
    "Q8_0": 2 + 32,
    "Q2_K": 256 // 16 + 256 // 4 + 2 + 2,
    "Q3_K": 256 // 8 + 256 // 4 + 12 + 2,
    "Q4_K": 2 + 2 + 12 + 256 // 2,
    "Q5_K": 2 + 2 + 12 + 256 // 8 + 256 // 2,
    "Q6_K": 256 // 2 + 256 // 4 + 256 // 16 + 2,
    "IQ4_XS": 2 + 2 + 256 // 2 + 256 // 64,
    "FP8": 1,
}

GGML_ELEMENTS_PER_BLOCK = {
    "F32": 1,
    "F16": 1,
    "BF16": 1,
    "Q4_0": 32,
    "Q5_0": 32,
    "Q8_0": 32,
    "Q2_K": 256,
    "Q3_K": 256,
    "Q4_K": 256,
    "Q5_K": 256,
    "Q6_K": 256,
    "IQ4_XS": 256,
    "FP8": 1,
}

DATA_TYPES = {
    "uint8": 0,
    "int8": 1,
    "uint16": 2,
    "int16": 3,
    "uint32": 4,
    "int32": 5,
    "float32": 6,
    "bool": 7,
    "string": 8,
    "array": 9,
    "uint64": 10,
    "int64": 11,
    "float64": 12,
    "FP8": 13,
}

kvalues_iq4nl = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], dtype=np.int8)

def dequantize_iq4_xs_cpu(data):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/21d3a308fcb7f31cb9beceaeebad4fb622f3c337/src/ggml-quants.c#L3568
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/21d3a308fcb7f31cb9beceaeebad4fb622f3c337/src/ggml-common.h#L393
    block_size = GGML_BLOCK_SIZES["IQ4_XS"]
    num_blocks = len(data) // block_size

    d = np.frombuffer(data, dtype=np.float16)[0::block_size//2].astype(np.float32).reshape(num_blocks, 1)
    scales_h = np.frombuffer(data, dtype=np.uint16)[1::block_size//2].reshape(num_blocks, 1)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, block_size)[:, 4:]
    scales_l = data_u8[:, :4].reshape(num_blocks, 4)
    qs = data_u8[:, 4:].reshape(num_blocks, block_size - 8)

    ls = np.zeros((num_blocks, QK_K // 32), dtype=np.int8)
    for ib in range(QK_K // 32):
        ls[:, ib] = ((scales_l[:, ib // 2] >> 4 * (ib % 2)) & 0xf) | (((scales_h[:, 0] >> 2 * ib) & 3) << 4)

    dl = (d * (ls - 32)).reshape(num_blocks, -1, 1)

    qs_lo_4 = qs[:, :QK_K // 2].reshape(num_blocks, -1, 16) & 0xf
    qs_hi_4 = qs[:, :QK_K // 2].reshape(num_blocks, -1, 16) >> 4

    y = np.zeros((num_blocks, QK_K), dtype=np.float32)
    for ib in range(QK_K // 32):
        y[:, ib*32:(ib*32)+16] = dl[:, ib] * kvalues_iq4nl[qs_lo_4[:, ib]]
        y[:, (ib*32)+16:(ib*32)+32] = dl[:, ib] * kvalues_iq4nl[qs_hi_4[:, ib]]

    return y.flatten()

def dequantize_iq4_xs_gpu(data: np.ndarray, device:str = "xpu", target_dtype = torch.get_default_dtype()):
    block_size = GGML_BLOCK_SIZES["IQ4_XS"]
    ele_per_blk = GGML_ELEMENTS_PER_BLOCK["IQ4_XS"]
    device = torch.device(device)
    num_blocks = len(data) // block_size
    data = np.frombuffer(data, dtype=data.dtype)
    c_pointer = ctypes.addressof(ctypes.cast(data.ctypes.data, ctypes.POINTER(ctypes.c_int8)).contents)
    return dequantize_extension.dequantize_iq4_xs(c_pointer, data.size, block_size, ele_per_blk, device, target_dtype)

def dequantize_q8_0(data):
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L43
    num_blocks = len(data) // GGML_BLOCK_SIZES["Q8_0"]

    scales = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, 1 + 16)[:, :1].astype(np.float32)
    qs = np.frombuffer(data, dtype=np.int8).reshape(num_blocks, 2 + 32)[:, 2:]
    return scales * qs

def dequantize_q8_0_gpu(data, device:str = "cuda", target_dtype = torch.get_default_dtype()):
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L43
    
    block_size = GGML_BLOCK_SIZES["Q8_0"]
    ele_per_blk = GGML_ELEMENTS_PER_BLOCK["Q8_0"]
    device = torch.device(device)
    data = np.frombuffer(data, dtype=data.dtype)
    c_pointer = ctypes.addressof(ctypes.cast(data.ctypes.data, ctypes.POINTER(ctypes.c_int8)).contents)
    return dequantize_extension.dequantize_q8_0(c_pointer, data.size, block_size, ele_per_blk, device, target_dtype)



ggml_name = "Q8_0"
num_blocks = 4
elements_per_block = GGML_ELEMENTS_PER_BLOCK[ggml_name]
block_size = GGML_BLOCK_SIZES[ggml_name]
size = block_size * elements_per_block * num_blocks


# Initialize the 1D np.ndarray with random data
data = np.random.randint(1, 256, size, dtype=np.uint8)

res_cpu = dequantize_q8_0(data)
res_cpu = torch.from_numpy(res_cpu)

res_gpu = dequantize_q8_0_gpu(data, device="xpu")

print(res_cpu.shape)
res_gpu = res_gpu.cpu().view(res_cpu.shape)
print(res_gpu.shape)
print(torch.allclose(res_gpu.cpu(), res_cpu))

print(res_cpu)
print(res_gpu)


# _, factors, offsets, qs1, qs2 = dequantize_iq4_xs_cpu(data)
# factors_cpu = torch.from_numpy(factors)
# offsets_cpu = torch.from_numpy(offsets)
# qs1_cpu = torch.from_numpy(qs1)
# qs2_cpu = torch.from_numpy(qs2)


# _, factors, offsets, qs1, qs2 = dequantize_iq4_xs_gpu(data, device="xpu")

# print(torch.allclose(factors.cpu(), factors_cpu))
# print(torch.allclose(offsets.cpu(), offsets_cpu))
# print(torch.allclose(qs1.cpu(), qs1_cpu))
# print(torch.allclose(qs2.cpu(), qs2_cpu))