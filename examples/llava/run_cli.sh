#!/bin/bash

Q="What is the address of this restaurant?"
# Q="Is this dine in or dine out receipt?"
# Q="What is the total amount paid?"
# Q="What is card holder's name?"
# Q="What is the transaction date?"
# Q="What is the phone number of this restaurant?"
# Q="Who is the attendant?"
# Q="Who is the cashier?"
# Q="Briefly describe this image."
prompt="<|system|>\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n<|user|>\n<image>\n $Q<|end|>\n<|assistant|>\n"
echo $prompt

# base_path=/export/share/yutong/xgenmm/llamacpp_wd/siglip_kosmos_phi3_4k_instruct/gguf
# # model=$base_path/phi3_mini_4k_instruct_f32.gguf
# model=$base_path/phi3_mini_4k_instruct_f16.gguf
# mmproj=$base_path/mmproj-model-f32.gguf

base_path=/export/share/yutong/xgenmm/llamacpp_wd/siglip_kosmos_phi3_4k_instruct_bf16_patch128/gguf
# model=$base_path/phi3_mini_4k_instruct_f16.gguf
model=$base_path/phi3_mini_4k_instruct_f16_Q4_K_M.gguf
mmproj=$base_path/mmproj-model-f32.gguf

./llama-llava-cli --model $model\
    --mmproj $mmproj \
    --image /export/share/yutong/receipt.jpg \
    --prompt "$prompt" \
    --seed 42 --ctx-size 4096 --predict 1024 \
    --temp 0.0 --ubatch-size 1280