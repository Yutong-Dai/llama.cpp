#!/bin/bash

# make xgenmm-cli

# IMG="/export/home/llama.cpp/examples/xgenmm/imgs/MathVista_MINI_1.png"
# Q="Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.\nQuestion: what is the total volume of the measuring cup? (Unit: g)"
# A="correct:1000 | minicpmv26:1000"

# IMG="/export/home/llama.cpp/examples/xgenmm/imgs/MathVista_MINI_4.png"
# Q="Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\nQuestion: Find $m\\angle H$\nChoices:\n(A) 97\n(B) 102\n(C) 107\n(D) 122"
# A="correct:A | minicpmv26:B"

IMG="/export/home/llama.cpp/examples/xgenmm/imgs/MathVista_MINI_52.png"
Q="Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\nQuestion: As shown in the figure, if CB = 4.0, DB = 7.0, and D is the midpoint of AC, then the length of AC is ()\nChoices:\n(A) 6cm\n(B) 7cm\n(C) 4cm\n(D) 5cm"
A="correct:A 6cm | minicpmv26:?"

# IMG="/export/home/llama.cpp/examples/xgenmm/imgs/MathVista_MINI_52.png"
# Q="Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.
# Question: What is the age gap between these two people in image? (Unit: years)"
# A="1"

# IMG="/export/home/llama.cpp/examples/xgenmm/imgs/MathVista_MINI_51.png"
# Q="Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.\nQuestion: Subtract all big red metallic spheres. Subtract all big brown matte things. How many objects are left?"
# A="correct:6 | xgemm:5"




model=/export/share/yutong/xgenmm/llamacpp_wd/minicpm26/ggml-model-Q4_K_M.gguf
mmproj=/export/share/yutong/xgenmm/llamacpp_wd/minicpm26/mmproj-model-f16.gguf

./build/bin/llama-minicpmv-cli --model $model\
    --mmproj $mmproj \
    --image $IMG\
    --prompt "$Q" \
    --seed 42 --ctx-size 4096 --predict 1024 \
    --temp 0.0 --verbose-prompt --color --ubatch-size 1280

echo "expected answer: $A"

# ./build/bin/llama-minicpmv-cli -m /export/share/yutong/xgenmm/llamacpp_wd/minicpm26/ggml-model-Q4_K_M.gguf --mmproj /export/share/yutong/xgenmm/llamacpp_wd/minicpm26/mmproj-model-f16.gguf -c 4096 --temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty 1.05 --image /export/home/llama.cpp/examples/xgenmm/imgs/receipt.jpg -p "What is the address of this restaurant?"    