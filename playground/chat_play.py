#!/usr/bin/env python3 
import os
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr


load_dotenv(override=True)
open_router_api_key = os.getenv('OPEN_ROUTER_API_KEY')

if open_router_api_key:
    print(f"OpenRouter API Key exists and begins {open_router_api_key[:8]}")
else:
    print("OpenRouter API Key not set")

# Models
OR_GPT = 'openai/gpt-oss-20b:free'
OR_DEEP_SEEK = 'deepseek/deepseek-chat-v3-0324:free'
OLLAMA = 'llama3.2:1b'

# API urls
OLLAMA_API = "http://localhost:11434/v1"
OR_API = "https://openrouter.ai/api/v1"

ollama = OpenAI(base_url=OLLAMA_API, api_key='ollama')

models = {
    'gpt-oss': OR_GPT,
    'llama3.2': OLLAMA,
    'deepseek-v3': OR_DEEP_SEEK
}

def get_model(model: str) -> str:
    try:
        result = models[model.lower()]
        return result
    except:
        raise ValueError("Unknown model")

system_message = "You are a helpful assistant"

def chat(message, history):
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]

    stream = ollama.chat.completions.create(model=OLLAMA, messages=messages, stream=True)

    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        yield response

chat = gr.ChatInterface(fn=chat, type="messages")
# chat.launch(share=True)
chat.launch()