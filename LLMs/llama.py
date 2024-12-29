import gc
import time
import json

from pathlib import Path
import sys
import os

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig, TextIteratorStreamer
from transformers.utils import get_json_schema
from threading import Thread

from function_calls import google_search_top_5, current_datetime

tools = [get_json_schema(google_search_top_5), get_json_schema(current_datetime)]

class Llama_3_8B():

    def __init__(self):
        self.history = self.init_history()
        self.reminder_prompt = """
            Remember, you are an ai assistant of the Aegean company providing customer support and answer only question that has to do with the company's info.
            if the user want to know something different answer kindly that you cant help with topic not relevand to aegean.
            Answer only in english, brief and clear. 
            If in order to answer to the user you need to call a function then respond only the JSON needed and nothing else!!
            Never reposnd the name of the function you call or here is the JSON format.
        """
        self.model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
        self.name = 'Llama 3.1 8B'
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
        # Set up the message to be tokenized
        self.add_to_history(role='system', prompt=self.reminder_prompt)
        tokenized_message = self.tokenizer.apply_chat_template(self.history, 
                                                               tools=tools, 
                                                               return_tensors="pt").to('cuda')

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

        function_call = False
        response = ''
        for i, text in enumerate(streamer):
            if i > 3:
                if str(text).startswith('{"name":') or str(text).startswith('{\n') or function_call:
                    function_call = True
                    response += text
                else:
                    yield text

        if function_call:
            for text in self.function_call(response):
                yield text

    def function_call(self, response):
        response = json.loads(response.strip())

        if response['name'] == 'google_search_top_5':
            results = 'The top searches I found are the following:\n' + '\n'.join(google_search_top_5(**response['parameters']))
        elif response['name'] == 'current_datetime':
            results = 'The current date and time is: ' + current_datetime()

        results = self.tokenizer.encode(results, add_special_tokens=False)
        for token in results:
            yield self.tokenizer.decode(token, skip_special_tokens=True)
