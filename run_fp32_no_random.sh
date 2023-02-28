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

export fused_softmax_with_triangular=False                                                                                  
export load_torch_random_345m_ckpt=True
export CUDA_VISIBLE_DEVICES=6 
export NVIDIA_TF32_OVERRIDE=0

source ../unset_paddle_env.sh 

rm -f my-gpt2_text/my-gpt2_text_document_gpt_* 

python tools/train.py \
    -c ppfleetx/configs/nlp/gpt/pretrain_gpt_345M_single_card.yaml \
    -o Engine.mix_precision.use_pure_fp16="False" \
    -o Engine.mix_precision.dtype="float16" \
    -o Engine.mix_precision.level="O0" \
    -o Engine.logging_freq="1" \
    -o Engine.save_load.save_steps="100000" \
    -o Optimizer.grad_clip.multi_precision=True \
    -o Data.Train.dataset.input_dir=./my-gpt2_text \
    -o Global.micro_batch_size=4 \
    -o Global.local_batch_size=4 \
    -o Global.seed=1234 \
    -o Global.global_batch_size=4 \
    -o Model.hidden_dropout_prob=0.0 \
    -o Model.attention_probs_dropout_prob=0.0 \
    -o Data.Eval.dataset.input_dir=./my-gpt2_text
