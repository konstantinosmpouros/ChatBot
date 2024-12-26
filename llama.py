import gc

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig, TextIteratorStreamer
from threading import Thread


class Llama_3_8B():
    
    def __init__(self):
        self.history = self.init_history()
        self.reminder_prompt = """
        Remember, you are an ai assistant of the Aegean company and answer only question that has to do with the company's info.
        if the user want to know something different answer kindly that you cant help with topic not relevand to aegean.
        Answer only in english, brief and clear.
        """
        self.model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
        self.tokenizer, self.model = None, None
    
    def init_history(self):
        system_message = """
            You are a helpful ai assistant that answer only in english of an airline company named aegean willing to help with anything that the user asks about the company and nothing different.
            You must always remain kind, and try first to understand what the user want and then respond.
        """

        messages = [
            {"role": "system", "content": system_message},
        ]
        return messages

    def add_to_history(self, role, prompt):
        self.history.append({'role': role, 'content': prompt})

    def load_model(self):
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token

            # Set quantization method
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"
            )

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, 
                                                              device_map="cuda", 
                                                              quantization_config=quant_config)

        except Exception as ex:
            print(ex.args)
            gc.collect()
            torch.cuda.empty_cache()
            return None, None

    def llm_response(self):
        # Set up the message to be tokenized
        self.add_to_history(role='system', prompt=self.reminder_prompt)
        tokenized_message = self.tokenizer.apply_chat_template(self.history, return_tensors="pt").to('cuda')

        # Initialize a stream to stream the response back
        streamer = TextIteratorStreamer(self.tokenizer, 
                                        skip_prompt=True,
                                        skip_special_tokens=True)

        # Generate response with in a thread
        generation_kwargs = {
            "input_ids": tokenized_message,  # Correctly pass input IDs
            "streamer": streamer,
            "temperature": 0.85,
            "max_new_tokens": 10000,
        }
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        self.add_to_history(role='assistant', prompt='')
        for i, text in enumerate(streamer):
            if i > 3:
                yield text
