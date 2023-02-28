# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def trans(torch_state, num_layer=24):
    torch_to_paddle = {}
    torch_to_paddle["gpt.embeddings.position_embeddings.weight"] = torch_state[
        "language_model"]["embedding"]["position_embeddings"]["weight"]
    torch_to_paddle["gpt.embeddings.word_embeddings.weight"] = torch_state[
        "language_model"]["embedding"]["word_embeddings"]["weight"]
    maps = []
    for i in range(num_layer):
        maps.append([
            f"gpt.decoder.layers.{i}.linear1.bias",
            f"layers.{i}.mlp.dense_h_to_4h.bias"
        ])
        maps.append([
            f"gpt.decoder.layers.{i}.linear1.weight",
            f"layers.{i}.mlp.dense_h_to_4h.weight", "T"
        ])
        maps.append([
            f"gpt.decoder.layers.{i}.linear2.bias",
            f"layers.{i}.mlp.dense_4h_to_h.bias"
        ])
        maps.append([
            f"gpt.decoder.layers.{i}.linear2.weight",
            f"layers.{i}.mlp.dense_4h_to_h.weight", "T"
        ])
        maps.append([
            f"gpt.decoder.layers.{i}.norm1.bias",
            f"layers.{i}.input_layernorm.bias"
        ])
        maps.append([
            f"gpt.decoder.layers.{i}.norm1.weight",
            f"layers.{i}.input_layernorm.weight"
        ])
        maps.append([
            f"gpt.decoder.layers.{i}.norm2.bias",
            f"layers.{i}.post_attention_layernorm.bias"
        ])
        maps.append([
            f"gpt.decoder.layers.{i}.norm2.weight",
            f"layers.{i}.post_attention_layernorm.weight"
        ])
        maps.append([
            f"gpt.decoder.layers.{i}.self_attn.out_proj.bias",
            f"layers.{i}.self_attention.dense.bias"
        ])
        maps.append([
            f"gpt.decoder.layers.{i}.self_attn.out_proj.weight",
            f"layers.{i}.self_attention.dense.weight", "T"
        ])
        maps.append([
            f"gpt.decoder.layers.{i}.self_attn.qkv_proj.bias",
            f"layers.{i}.self_attention.query_key_value.bias"
        ])
        maps.append([
            f"gpt.decoder.layers.{i}.self_attn.qkv_proj.weight",
            f"layers.{i}.self_attention.query_key_value.weight", "T"
        ])

    maps.append([f"gpt.decoder.norm.weight", f"final_layernorm.weight"])
    maps.append([f"gpt.decoder.norm.bias", f"final_layernorm.bias"])

    for m in maps:
        if len(m) == 2:
            torch_to_paddle[m[0]] = torch_state["language_model"]["encoder"][m[
                1]]
        else:
            torch_to_paddle[m[0]] = torch_state["language_model"]["encoder"][m[
                1]].T

    for k in torch_to_paddle.keys():
        torch_to_paddle[k] = torch_to_paddle[k].numpy()

    return torch_to_paddle


if __name__ == "__main__":
    import torch
    news = torch.load(
        '/root/paddlejob/workspace/env_run/gpt_benchmark/Megatron-LM/ckpt_345m_init.bin',
        map_location="cpu")
    paddle = trans(news)
    pass
