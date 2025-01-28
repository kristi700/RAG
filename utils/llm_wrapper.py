import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PRIVATE_TOKEN = ''

class LLM_wrapper():
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=PRIVATE_TOKEN)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto", token=PRIVATE_TOKEN)

    def generate(self, user_prompt: str, context):
        system_prompt = f"""You are a helpful assistant. You answer the user's question based on the context provided here.
        Do not make up data, ground your answers in the context.
        The context contains a set of document chunks, knowledge graph nodes and knowledge graph connections. Use these to answer the question.
        Do not mention this knowledge graph directly, use fluent natural language instead.
        Use English regardless of the input data.

        ### Context ###
        {context}"""

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
