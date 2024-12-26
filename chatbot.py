import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import login

import gc

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig, TextIteratorStreamer
from threading import Thread


def hf_login():
    # Load .env file and pass the keys to os as env variables
    load_dotenv()
    token = os.getenv('HUGGINGFACE_TOKEN')

    # Log in to hugging face
    if token:
        login(token=token)
    else:
        raise ValueError("HUGGINGFACE_TOKEN is not set. Please ensure the .env file exists and contains the token. Else please create an enviroment variable named 'HUGGINGFACE_TOKEN' and place your token inside.")

def init_prompts():
    system_message = """
        You are a helpful ai assistant that answer only in english of an airline company named aegean willing to help with anything that the user asks about the company and nothing different.
        You must always remain kind, and try first to understand what the user want and then respond.
    """

    messages = [
        {"role": "system", "content": system_message},
    ]
    return messages

def add_to_history(history, role, prompt):
    history.append({'role': role, 'content': prompt})
    return history

def add_remider():
    prompt = """
    Remember, you are an ai assistant of the Aegean company and answer only question that has to do with the company's info.
    if the user want to know something different answer kindly that you cant help with topic not relevand to aegean.
    Answer only in english, brief and clear.
    """
    return prompt

def load_model(model_name):
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if tokenizer.eos_token is None:
            tokenizer.eos_token = "<|endoftext|>"
        tokenizer.pad_token = tokenizer.eos_token

        # Set quantization method
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )

        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     device_map="cuda",
                                                     quantization_config=quant_config)

        return tokenizer, model

    except Exception as ex:
        print(ex.args)
        gc.collect()
        torch.cuda.empty_cache()
        return None, None

def llm_response(tokenizer, model, history):
    # Set up the message to be tokenized
    add_to_history(history, role='system', prompt=add_remider())
    tokenized_message = tokenizer.apply_chat_template(history, return_tensors="pt").to('cuda')

    # Initialize a stream to stream the response back
    streamer = TextIteratorStreamer(tokenizer, 
                                    skip_prompt=True,
                                    skip_special_tokens=True)

    # Generate response with in a thread
    generation_kwargs = {
        "input_ids": tokenized_message,  # Correctly pass input IDs
        "streamer": streamer,
        "temperature": 0.85,
        "max_new_tokens": 10000,
    }
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for i, text in enumerate(streamer):
        if i > 3:
            yield text

