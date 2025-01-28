import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PRIVATE_TOKEN = ''

class LLM_wrapper():
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=PRIVATE_TOKEN)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto", token=PRIVATE_TOKEN)

    def generate(self, system_prompt: str, user_prompt: str):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
