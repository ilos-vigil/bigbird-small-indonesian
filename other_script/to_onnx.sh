#!/bin/bash
# You may need to change value of `--atol`

python -m transformers.onnx --model=./checkpoint-model-bigbird-small-indonesian/checkpoint-6450 --feature masked-lm --preprocessor tokenizer --atol 1e-5 ./onnx_model