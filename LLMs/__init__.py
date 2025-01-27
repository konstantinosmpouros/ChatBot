import gc
import json
import copy

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

from function_calls import rag_retrieve, google_search_top_5, current_datetime
from RAG import RAGPipeline

tools = [
    get_json_schema(rag_retrieve),
    get_json_schema(google_search_top_5),
    get_json_schema(current_datetime),
]

class Llama_3_8B():

    def __init__(self):
        # Initialize history
        self.history = self.init_history()
        
        # Initialize reminder prompt
        self.reminder_prompt = """
            Remember:
            - Your role is to assist users by any information they need and for people and resumes from the RAG db.
            - When the user asks about Attica Group, a person, a CV, skills, experience, job positions or related topics, **always use the `rag_retrieve` function** to fetch information in this format:
                {"name": "rag_retrieve","parameters": {"query": "<user query>"}}
            - If you find any documents in the system prompt, summarize the retrieved content in order to answer in the user query appropriately.
            - Never answer with the JSON when the documents have been retrieved from the knowledge base.
            - Never answer something for a person if you dont see documents in the system prompt.
            - When summarizing or presenting information retrieved from documents or external sources instead of saying 'Based on retrieved chunks,' phrase your response like 'From what I've found...' or 'Here’s what I learned...,' adapting your phrasing to make the interaction feel human and smooth.
            - When summarizing documents, rephrase key points in a manner that directly addresses the user's query while maintaining a friendly tone.
        """

        self.model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
        self.name = 'Llama 3.1 8B'
        self.tokenizer, self.model = None, None

        # Initialize RAG pipeline
        self.rag = RAGPipeline()

    def init_history(self):
        system_message = """
            You are a friendly and helpful AI assistant with a RAG to help the user with his queries. Your primary responsibilities are:

            1. Respond to user kindly and friendly
            2. When the user asks about Attica Group, person, a resume or persons with certain skill, job position or experience retrieve info from the RAG database using the following function call format:
                {"name": "rag_retrieve", "parameters": {"query": "<user query>"}}
            3. If documents are already retrieved, respond to the user queries by summarizing the retrieved documents relevant to the user's query in the system prompt.
            4. When your answer refers to the RAG retrieved content, your answer should be natural. Dont start with "Based on the retrieved documents,".
            5. Never answer with the JSON when the documents have been retrieved from the knowledge base.
            6. Never answer something for a person that the info doesn't come from the retrieved documents.
            7. If you have already retrieved documents and they are available, do not call the `rag_retrieve` function again for the same query unless explicitly requested.
            8. Always validate the availability of retrieved documents before responding about a person, CV, or related topic.
            9. When summarizing or presenting information retrieved from documents or external sources instead of saying 'Based on retrieved chunks,' phrase your response like 'From what I've found...' or 'Here’s what I learned...,' adapting your phrasing to make the interaction feel human and smooth.
            
            Your role is critical for ensuring accurate and reliable answers based on the information stored in the RAG database. At the start of the conversation greet warm the user.
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
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        except Exception as ex:
            print(ex.args)
            gc.collect()
            torch.cuda.empty_cache()
            return None, None

    def llm_response(self):
        # Add the reminder prompt
        self.add_to_history(role='system', prompt=self.reminder_prompt)

        # Generate the response according to the user query
        streamer = self.generate(history=self.history)

        # Add in the history a new line to store the answer
        self.add_to_history(role='assistant', prompt='')

        function_call = False
        response = ''
        for i, text in enumerate(streamer):
            if i > 3:
                if str(text).strip().startswith('{"name"') or str(text).strip().startswith('{') or function_call:
                    function_call = True
                    response += text
                else:
                    yield text

        if function_call:
            print(response)
            for text in self.function_call(response):
                yield text

    def generate(self, history):
        tokenized_message = self.tokenizer.apply_chat_template(history, 
                                                               tools=tools, 
                                                               return_tensors="pt").to('cuda')
        attention_mask = torch.ones(tokenized_message.shape, device='cuda')

        # Initialize a stream to stream the response back
        streamer = TextIteratorStreamer(self.tokenizer, 
                                        skip_prompt=True,
                                        skip_special_tokens=True)

        # Generate response with in a thread
        generation_kwargs = {
            "input_ids": tokenized_message,  # Correctly pass input IDs
            "attention_mask": attention_mask,
            "streamer": streamer,
            "temperature": 0.85,
            "max_new_tokens": 1000,
        }
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        return streamer

    def function_call(self, response):
        response = json.loads(response.strip())

        if response['name'] == 'rag_retrieve':
            retrieved_chunks = self.rag.retrieve(**response['parameters'])
            cloned_history = copy.deepcopy(self.history[:-1])
            cloned_history.append({'role': 'user', 'content': retrieved_chunks})
            streamer = self.generate(cloned_history)
            for i, text in enumerate(streamer):
                if i > 3:
                    yield text
