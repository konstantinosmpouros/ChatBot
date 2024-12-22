import os
import json

from dotenv import load_dotenv
from huggingface_hub import login
import gc

from openai import OpenAI

import gradio as gr

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

def hf_login():
    # Load .env file and pass the keys to os as env variables
    load_dotenv()
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
    os.environ['HUGGINGFACE_TOKEN'] = os.getenv('HUGGINGFACE_TOKEN')

    # Log in to hugging face
    login(token=os.getenv('HUGGINGFACE_TOKEN'))


def init_history():
    system_message = """
        You are a helpful assistant willing to help with anything that the user asks.
    """

    messages = [
        {"role": "system", "content": system_message},
    ]
    return messages


def add_to_history(history, role, prompt):
    history.append({'role': role, 'content': prompt})
    return history


def load_llm(model_name):
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Set quantization method
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     device_map="cuda",
                                                     quantization_config=quant_config)
    
        return tokenizer, model
        
    except Exception as ex:
        print(ex.args)
        return None, None
        
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def chat(tokenizer, model, history, prompt):
    # Set up the message to be tokenized
    history = add_to_history(history, role='user', prompt=prompt)
    tokenized_message = tokenizer.apply_chat_template(history, return_tensors="pt").to('cuda')

    # Generate response
    response = model.generate(tokenized_message, max_new_tokens=1000)
    response = response[0][len(tokenized_prompt[0]):]

    # Decode response
    output = tokenizer.decode(response, skip_special_tokens=True)
    output = output.split('\n\n')[1]
    history = add_to_history(history, role='assistant', prompt=output)



