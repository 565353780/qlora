import logging

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer, TextGenerationPipeline

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)

pretrained_model_dir = \
    "/home/chli/github/textgen/models/llama-2-7b-chat-hf/"
quantized_model_dir = \
    "/home/chli/github/textgen/models/llama-2-7b-chat-hf-GPTQ/"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
examples = [
    tokenizer(
        "auto-gptq is an easy-to-use model quantization library " +
        "with user-friendly apis, based on GPTQ algorithm."
    )
]

quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=True,
)

model = AutoGPTQForCausalLM.from_pretrained(
    pretrained_model_dir, quantize_config)

model.quantize(examples)

model.save_quantized(quantized_model_dir)

model.save_quantized(quantized_model_dir, use_safetensors=True)

model = AutoGPTQForCausalLM.from_quantized(
    quantized_model_dir, device="cuda:0")

print(tokenizer.decode(model.generate(
    **tokenizer("auto_gptq is", return_tensors="pt").to(model.device))[0]))

pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
print(pipeline("auto-gptq is")[0]["generated_text"])
