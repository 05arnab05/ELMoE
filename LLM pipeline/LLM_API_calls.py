# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 10:03:44 2025

@author: arnab
"""
import os
from openai import OpenAI
from huggingface_hub import InferenceClient
import tiktoken
from dotenv import load_dotenv

from together import Together as TogetherClient

# Load environment variables from a .env file
load_dotenv()


def count_tokens(text):
    """Counts the number of tokens in the text."""
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def LLama(role, content):
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    if not HUGGINGFACE_TOKEN:
        raise ValueError(
            "Missing Hugging Face API token in environment variables.")

    client = InferenceClient(
        provider="together",
        api_key=HUGGINGFACE_TOKEN,
    )

    completion = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct",
        messages=[
            {'role': 'system', 'content': role},
            {'role': 'user', 'content': content}
        ],
        max_tokens=500,
    )
    return completion.choices[0].message.content


def Together_models(role, content, llm_model_name):
    together_models = {
        "Llama400b": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",

        "Qwen32b": "Qwen/Qwen3-VL-32B-Instruct",


        "Mistral14b": "mistralai/Ministral-3-14B-Instruct-2512",

        "Llama8b":"meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "Llama3b": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
        "Qwen1.7b": "Qwen/Qwen2.5-1.7B-Instruct",
        "Qwen0.6b": "Qwen/Qwen2.5-0.5B-Instruct"
    }
    
    llm_name_normalized = llm_model_name.lower()
    together_models_lower = {k.lower(): v for k, v in together_models.items()}
    
    if llm_name_normalized in together_models_lower:
        together_model_id = together_models_lower[llm_name_normalized]
    
        result=Together_LLM_API_call(role, content, together_model_id)
        return result
    else:
        return False
    
    

def Together_LLM_API_call(role, content, llm_model_id):

    client = TogetherClient()  # uses TOGETHER_API_KEY from env

    response = client.chat.completions.create(
        model=llm_model_id, #"Qwen/Qwen3-VL-32B-Instruct"
        messages=[
            {"role": "system", "content": role},
            {"role": "user", "content": content}
        ],
        max_tokens=500,
    )

    return response.choices[0].message.content


def ChatGPT(role, content):
    API_KEY = os.getenv("OPENAI_API_KEY")
    if not API_KEY:
        raise ValueError("Missing OpenAI API key in environment variables.")

    client = OpenAI(api_key=API_KEY)

    response = client.chat.completions.create(
        messages=[
            {'role': 'system', 'content': role},
            {'role': 'user', 'content': content}
        ],
        model="gpt-4o",
        temperature=0.7,
        max_tokens=1000
    )

    return response.choices[0].message.content