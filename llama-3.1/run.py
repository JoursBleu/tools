import time
import torch
import transformers

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


model = AutoModelForCausalLM.from_pretrained(
            "/mnt/volumes/lpai-demo-muses/lt/models/Meta-Llama-3.1-405B-Instruct",
            use_flash_attention_2=False,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True,
        )

tokenizer = AutoTokenizer.from_pretrained("/mnt/volumes/lpai-demo-muses/lt/models/Meta-Llama-3.1-405B-Instruct",)

messages = [
    {"role": "system", "content": "You are a bot that responds to any question."},
    {"role": "user", "content": "Born in north-east France, Soyer trained as a chef, introduce him."}
]
curr_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,)

input_ids = tokenizer.encode(curr_text, return_tensors='pt')

output = model.generate(input_ids, max_new_tokens=128)
print(output)

start = time.time()
for i in range(10):
    output = model.generate(input_ids, max_new_tokens=128)
end = time.time()

print("Time:", (end-start)/(i+1), "s")
breakpoint()
