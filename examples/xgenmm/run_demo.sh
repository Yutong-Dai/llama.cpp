# Q="Who is the cashier?"
# image_path="/export/home/llama.cpp/examples/xgenmm/imgs/receipt.jpg"

# Q="Read this image and convert it to the dict of the form {product name: the price under Grand total}."
# image_path="examples/xgenmm/imgs/demos/table.png"

# Q="Extract the information in JSON format. Include name, dob, address, and expiration date."
# image_path="/export/home/llama.cpp/examples/xgenmm/imgs/demos/driverslicense.jpg"

# Q="Convert the table to JSON format. Include year and Employees."
# image_path="/export/home/llama.cpp/examples/xgenmm/imgs/demos/table-mm.jpg"

Q="Convert the table to markdown code."
image_path="/export/home/llama.cpp/examples/xgenmm/imgs/demos/table-mm.jpg"

prompt="<|system|>\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n<|user|>\n<image>\n $Q<|end|>\n<|assistant|>\n"
echo $prompt

base_path=/export/share/yutong/xgenmm/llamacpp_wd/siglip_kosmos_phi3_4k_instruct_bf16_patch128/gguf
model=$base_path/phi3_mini_4k_instruct_f16_Q4_K_M.gguf
mmproj=$base_path/mmproj-model-f16.gguf

./xgenmm-cli --model $model\
    --mmproj $mmproj \
    --image $image_path\
    --prompt "$prompt" \
    --seed 42 --ctx-size 4096 --predict 1024 \
    --temp 0.0 --verbose-prompt --color --ubatch-size 1280