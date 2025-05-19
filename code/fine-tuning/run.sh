# preprocess datasets - optional but recommended
CUDA_VISIBLE_DEVICES="0" axolotl preprocess examples/phi/lora-3.5.yaml


### working config with example data
conda activate unsloth
cd llm-reason-hg/axolotl

CUDA_VISIBLE_DEVICES="0" axolotl preprocess examples/phi/phi3-lora-HypER.yaml

# finetune lora phi3

accelerate launch -m axolotl.cli.train examples/phi/phi3-lora-HypER.yaml
CUDA_VISIBLE_DEVICES="0" python -m axolotl.cli.inference examples/phi/phi3-lora-HypER.yaml --lora_model_dir="./outputs/phi3-hypER1-lora-out"

accelerate launch -m axolotl.cli.train examples/phi/phi3-lora-HypER_multihop12.yaml
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8" python3 -m axolotl.cli.merge_lora examples/phi/phi3-lora-HypER_multihop12.yaml --lora_model_dir="./outputs/phi3-hypER-mixed-lora-out-full"

#### end of working commands ########
#### Llama 3.2

accelerate launch -m axolotl.cli.train examples/llama-3/llama3.2-lora-HypER.yml
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8" python3 -m axolotl.cli.merge_lora examples/llama-3/llama3.2-lora-HypER.yml --lora_model_dir="./outputs/llama3.2-3B-hypER-mixed-lora-out-full"



###  long llama 3b try #########

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8" axolotl train examples/openllama-3b/long-llama-lora-HypER.yml

accelerate launch -m axolotl.cli.train examples/openllama-3b/long-llama-lora-HypER.yml

#### mistralLite

accelerate launch -m axolotl.cli.train examples/mistral/mistralLite-lora-HypER.yml

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8" python3 -m axolotl.cli.merge_lora examples/mistral/mistralLite-lora-HypER.yml --lora_model_dir="./outputs/mistralLite-hypER-mixed-lora-out-full2"


##############################


accelerate launch -m axolotl.cli.train examples/phi/lora-3.5.yaml --deepspeed deepspeed_configs/zero1.json
accelerate launch -m axolotl.cli.train examples/openllama-3b/lora.yml --deepspeed deepspeed_configs/zero1.json

# inference
axolotl inference examples/phi/lora-3.5.yaml \
    --lora-model-dir="./outputs/lora-out"

# gradio
axolotl inference examples/phi/lora-3.5.yaml \
    --lora-model-dir="./outputs/lora-out" --gradio

# remote yaml files - the yaml config can be hosted on a public URL
# Note: the yaml config must directly link to the **raw** yaml
axolotl train https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/main/examples/llama-3/lora-1b.yml