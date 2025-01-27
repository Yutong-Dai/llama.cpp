#!/bin/bash

make xgenmm-cli

# for xgenmm

# IMG="/export/home/llama.cpp/examples/xgenmm/imgs/MathVista_MINI_52.png"
# Q="Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.
# Question: What is the age gap between these two people in image? (Unit: years)"
# A="1"

# IMG="/export/home/llama.cpp/examples/xgenmm/imgs/MathVista_MINI_51.png"
# Q="Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.\nQuestion: Subtract all big red metallic spheres. Subtract all big brown matte things. How many objects are left?"
# A="correct:6 | xgemm:5"

# IMG="/export/home/llama.cpp/examples/xgenmm/imgs/MathVista_MINI_1.png"
# Q="Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.\nQuestion: what is the total volume of the measuring cup? (Unit: g)"
# A="correct:1000 | xgemm:1000"

prompt="<|system|>\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n<|user|>\n<image>\n $Q<|end|>\n<|assistant|>\n"
echo $prompt



base_path=/export/share/yutong/xgenmm/llamacpp_wd/siglip_kosmos_phi3_4k_instruct_bf16_patch128/gguf
# base_path=/export/share/yutong/xgenmm/llamacpp_wd/siglip_kosmos_phi3_4k_instruct_pre_V3_6-v3.8.0/gguf
# model=$base_path/phi3_mini_4k_instruct_f16.gguf
model=$base_path/phi3_mini_4k_instruct_f16_Q4_K_M.gguf
mmproj=$base_path/mmproj-model-f16.gguf

./xgenmm-cli --model $model\
    --mmproj $mmproj \
    --image $IMG\
    --prompt "$prompt" \
    --seed 42 --ctx-size 4096 --predict 1024 \
    --temp 0.0 --verbose-prompt --color --ubatch-size 1280

echo "expected answer: $A"

# ./bin/llama-minicpmv-cli -m /export/share/yutong/xgenmm/llamacpp_wd/minicpm26/ggml-model-Q4_K_M.gguf --mmproj /export/share/yutong/xgenmm/llamacpp_wd/minicpm26/mmproj-model-f16.gguf -c 4096 --temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty 1.05 --image /export/home/llama.cpp/examples/xgenmm/imgs/receipt.jpg -p "What is the address of this restaurant?"    