import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer

model_dir = "./llama-2-7b-chat-hf"

model = LlamaForCausalLM.from_pretrained(model_dir, device_map=None).to("cpu")
tokenizer = LlamaTokenizer.from_pretrained(model_dir)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float32,
    device=-1,
)

prompt = 'I have tomatoes, basil and cheese at home. What can I cook for dinner?\n' 

sequences = pipeline(
    prompt,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=50,
)

for seq in sequences:
    print(f"{seq['generated_text']}")
