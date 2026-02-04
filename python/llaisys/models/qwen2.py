from typing import Sequence
from ..libllaisys import LIB_LLAISYS, DeviceType, DataType, llaisysTensor_t, llaisysDataType_t, llaisysDeviceType_t
from ..libllaisys import LlaisysQwen2Meta, LlaisysQwen2Model_p, LlaisysQwen2Weights_p
from ..tensor import Tensor

from pathlib import Path
import safetensors
import numpy as np
import ctypes
import json
import os
import torch # Add import

class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        self.device = device
        
        args_path = model_path / "config.json"
        with open(args_path, "r") as f:
            cfg = json.load(f)

        self.meta = LlaisysQwen2Meta()
        self.meta.dtype = DataType.F32.value

        self.meta.nlayer = cfg["num_hidden_layers"]
        self.meta.hs = cfg["hidden_size"]
        self.meta.nh = cfg["num_attention_heads"]
        self.meta.nkvh = cfg["num_key_value_heads"]
        self.meta.dh = self.meta.hs // self.meta.nh
        self.meta.di = cfg["intermediate_size"]
        self.meta.maxseq = 2048
        self.meta.voc = cfg["vocab_size"]
        self.meta.epsilon = cfg["rms_norm_eps"]
        self.meta.theta = cfg.get("rope_theta", 10000.0) or 10000.0
        self.meta.end_token = 151643 

        self.model_ptr = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(self.meta),
            llaisysDeviceType_t(device.value),
            None, 0
        )
        
        weights_struct = LIB_LLAISYS.llaisysQwen2ModelWeights(self.model_ptr).contents
        
        print("\nLoading weights...")
        files = sorted(list(model_path.glob("*.safetensors")))
        for file in files:
            print(f"  - {file.name}")
            # Use 'pt' framework to handle bf16
            with safetensors.safe_open(file, framework="pt", device="cpu") as f:
                for k in f.keys():
                    pt_tensor = f.get_tensor(k)
                    # Convert to F32 Numpy
                    data = pt_tensor.to(torch.float32).detach().numpy()

                    dest_tensor_ptr = None
                    
                    if k == "model.embed_tokens.weight":
                        dest_tensor_ptr = weights_struct.in_embed
                    elif k == "lm_head.weight":
                        dest_tensor_ptr = weights_struct.out_embed
                    elif k == "model.norm.weight":
                        dest_tensor_ptr = weights_struct.out_norm_w
                    elif k.startswith("model.layers."):
                        parts = k.split(".")
                        idx = int(parts[2])
                        suffix = ".".join(parts[3:])
                        
                        if suffix == "input_layernorm.weight": dest_tensor_ptr = weights_struct.attn_norm_w[idx]
                        elif suffix == "self_attn.q_proj.weight": dest_tensor_ptr = weights_struct.attn_q_w[idx]
                        elif suffix == "self_attn.q_proj.bias": dest_tensor_ptr = weights_struct.attn_q_b[idx]
                        elif suffix == "self_attn.k_proj.weight": dest_tensor_ptr = weights_struct.attn_k_w[idx]
                        elif suffix == "self_attn.k_proj.bias": dest_tensor_ptr = weights_struct.attn_k_b[idx]
                        elif suffix == "self_attn.v_proj.weight": dest_tensor_ptr = weights_struct.attn_v_w[idx]
                        elif suffix == "self_attn.v_proj.bias": dest_tensor_ptr = weights_struct.attn_v_b[idx]
                        elif suffix == "self_attn.o_proj.weight": dest_tensor_ptr = weights_struct.attn_o_w[idx]
                        elif suffix == "post_attention_layernorm.weight": dest_tensor_ptr = weights_struct.mlp_norm_w[idx]
                        elif suffix == "mlp.gate_proj.weight": dest_tensor_ptr = weights_struct.mlp_gate_w[idx]
                        elif suffix == "mlp.up_proj.weight": dest_tensor_ptr = weights_struct.mlp_up_w[idx]
                        elif suffix == "mlp.down_proj.weight": dest_tensor_ptr = weights_struct.mlp_down_w[idx]
                    
                    if dest_tensor_ptr:
                        LIB_LLAISYS.tensorLoad(dest_tensor_ptr, data.ctypes.data_as(ctypes.c_void_p))

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        if max_new_tokens is None: max_new_tokens = 50
        
        generated = []
        current_ctx = list(inputs)
        
        for _ in range(max_new_tokens):
            if len(generated) == 0:
                feed = current_ctx
            else:
                feed = [generated[-1]]
            
            arr = (ctypes.c_int64 * len(feed))(*feed)
            
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self.model_ptr,
                arr,
                ctypes.c_size_t(len(feed))
            )
            
            # 修改点 1: 先保存 token，再检查是否结束，确保 EOS 被包含
            generated.append(next_token)
            
            if next_token == self.meta.end_token:
                break

        # 修改点 2: 返回完整的序列 (输入 + 生成)，与 HuggingFace 的输出对其
        return list(inputs) + generated
    
    def __del__(self):
        if hasattr(self, "model_ptr"):
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self.model_ptr)