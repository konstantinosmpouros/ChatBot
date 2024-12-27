import gc
import json
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig, TextIteratorStreamer
from transformers.utils import get_json_schema
from threading import Thread

from tools import google_search_top_5, current_datetime

tools = [get_json_schema(google_search_top_5), get_json_schema(current_datetime)]

class Mistral_7B():

    def __init__(self):
        self.history = self.init_history()
        self.reminder_prompt = """
            Remember, you are an ai assistant of the Aegean company providing customer support and answer only question that has to do with the company's info.
            if the user want to know something different answer kindly that you cant help with topic not relevand to aegean.
            Answer only in english, brief and clear. 
            If in order to answer to the user you need to call a function then respond only the JSON needed and nothing else!!
        """
        self.model_name = 'mistralai/Mistral-7B-Instruct-v0.3'
        self.name = 'Mistral 7B v0.3'
        self.tokenizer, self.model = None, None

    def init_history(self):
        system_message = """
            You are a helpfull ai assistant for an airline company named Aegean. Your tasks are the following:
                1. Provide customer service support to the user about Aegean policies, flights, services etc.
                2. Answer only in english, brief and clear, understanding first what the user needs.
                3. Be always polity and kind with any user! Always remember that you are made for customer support.
                4. If the user asks you to answer about something non related to aegean company and the relevant customer support, answer kindly that you cant answer that.
                5. If the user want you to execute a function call, answer only the JSON format and nothing more!! Nothing more!!
            When starting the conversation, greet kindly the user and then proceed to the customer support.
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

    def unload_model(self):
        del self.tokenizer, self.model
        gc.collect()
        torch.cuda.empty_cache()
        self.tokenizer, self.model = None, None
        time.sleep(3)

    def llm_response(self):
        pass

