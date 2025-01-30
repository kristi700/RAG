import torch
import outlines

from outlines import generate
from transformers import AutoTokenizer, AutoModelForCausalLM

PRIVATE_TOKEN = ''
DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. You answer the user's question based on the context provided here.
        Do not make up data, ground your answers in the context.
        The context contains a set of document chunks, knowledge graph nodes and knowledge graph connections. Use these to answer the question.
        Do not mention this knowledge graph directly, use fluent natural language instead.
        Use English regardless of the input data.

        ### Context ###
        """

DEFAULT_TRIPLET_EXTRACTION_PROMPT = """Given a text chunk from a document extract up to 5 and not more knowledge triplets in the form of (subject, predicate, object).
    Stopwords should be avoided. Provide the triples as a JSON object along with a short description of the subject and object entities. Descriptions could be longer than the example provided.
    Use English regardless of the input data. Make sure not to extract more than 5 triplets!

    You must output a strictly valid JSON object. Any output that is not valid JSON will be rejected.
    Ensure that:
    - No extra commas appear before closing brackets (`]` or `}`).
    - Every `{` has a matching `}`.
    - Strings are wrapped in double quotes (`"`), not single quotes (`'`).
    - Follow this exact JSON format:

    Example:
    Text: Philz is a coffee shop founded in 1982 in Berkeley. The café specializes in handcrafted coffee.

    Response example:
{
    "triplets": [
        {
            "subject": {"name": "Philz", "description": "The name of a coffee shop"},
            "predicate": "type of business",
            "object": {"name": "coffee shop", "description": "A place specializing in serving coffee"}
        },
        {
            "subject": {"name": "Philz", "description": "The name of a coffee shop"},
            "predicate": "located in",
            "object": {"name": "Berkeley", "description": "A city in California"}
        },
        {
            "subject": {"name": "Philz", "description": "The name of a coffee shop"},
            "predicate": "founded in",
            "object": {"name": "1982", "description": "A year, date"}
        },
        {
            "subject": {"name": "Philz", "description": "The name of a coffee shop"},
            "predicate": "specializes in",
            "object": {"name": "handcrafted coffee", "description": "Coffee made by hand, generally of high quality"}
        }
    ]
}
"""

OUTLINES_SCHEMA = """
{
    "triplets": [
        {
            "subject": {"name": "Philz", "description": "The name of a coffee shop"},
            "predicate": "type of business",
            "object": {"name": "coffee shop", "description": "A place specializing in serving coffee"}
        },
        {
            "subject": {"name": "Philz", "description": "The name of a coffee shop"},
            "predicate": "located in",
            "object": {"name": "Berkeley", "description": "A city in California"}
        },
        {
            "subject": {"name": "Philz", "description": "The name of a coffee shop"},
            "predicate": "founded in",
            "object": {"name": "1982", "description": "A year, date"}
        },
        {
            "subject": {"name": "Philz", "description": "The name of a coffee shop"},
            "predicate": "specializes in",
            "object": {"name": "handcrafted coffee", "description": "Coffee made by hand, generally of high quality"}
        }
    ]
}

"""

class LLM_wrapper():
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        torch.set_num_threads(8) # NOTE - doesnt seem like this works ngl
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=PRIVATE_TOKEN)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto", token=PRIVATE_TOKEN)
        #self.outlines_model = outlines.models.transformers(model_name)

    def generate_chat(self, user_prompt: str, context, system_prompt: str = DEFAULT_SYSTEM_PROMPT ):
        # TODO doublecheck!
        combined_prompt = system_prompt + context
        
        messages = [
            {"role": "system", "content": combined_prompt},
            {"role": "user", "content": user_prompt}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True # NOTE - do we not need this here?
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=512)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    def generate_extract(self, data, extraction_prompt: str = DEFAULT_TRIPLET_EXTRACTION_PROMPT):
        messages = [{"role": "system", "content": extraction_prompt},
                    {"role": "user", "content": data}
                    ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) 
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=10000)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        