# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Official Qwen/Qwen3-VL-2B-Instruct (HF revision 89644892e4d85e24eaac8bacfd4f463576704203).

MODEL_ARGS=(
   --swiglu
   --num-layers 28
   --hidden-size 2048
   --ffn-hidden-size 6144
   --num-attention-heads 16
   --group-query-attention
   --num-query-groups 8
   --use-rotary-position-embeddings
   --disable-bias-linear
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   --rotary-base 5000000
   --vocab-size 151936
   --kv-channels 128
   --qk-layernorm
)
