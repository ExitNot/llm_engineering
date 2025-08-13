#!/usr/bin/env python3 
import os
from dotenv import load_dotenv
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
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

DEFAULT_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"}

system_prompt = "You are a technical expert with a huge amount of experience. \
    Your main task is to summarize the article from the webpage, ignoring text that might be navigation-related. \
    Additionally, provide 1-3 quotes that hold the main theses of the article. \
    Respond in Markdown. Do not repeate your task in your answer"

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

class Article:
    url: str
    title: str
    text: str

    def __init__(self, url: str, headers = DEFAULT_HEADERS):
        """
        Create this Article object from the given url with extracted title and links using the BeautifulSoup library
        """
        self.url = url
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        title_element = soup.find('h1', class_='story-title')
        self.title = title_element.get_text(strip=True) if title_element else "No title found"        
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)

    def _user_prompt(self) -> str:
        user_prompt = f"You are looking at an article with title: {self.title}"
        user_prompt += "\nThe contents of this website is as follows; \
            please provide an exhaustive but medium size summary of this article in markdown. \
            Start your answer with article title that stored in .\n\n"
        user_prompt += self.text
        return user_prompt
    
    def _build_msg(self):
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self._user_prompt()}
        ]
    
    def summarize(self, model: str):
        stream = ollama.chat.completions.create(
            model = get_model(model),
            messages = self._build_msg(),
            stream = True
        )
        result = ""
        for chunk in stream:
            result += chunk.choices[0].delta.content or ""
            yield result


def stream_article_summary(url: str, model: str):
    yield ""
    article = Article(url)
    yield from article.summarize(model)

view = gr.Interface(
    fn=stream_article_summary,
    inputs=[
        gr.Textbox(label="Article url (only public urls):"),
        gr.Dropdown(["Llama3.2", "GPT-oss", "DeepSeek-v3"], label="Select model")],
    outputs=[gr.Markdown(label="Article summary:")],
    flagging_mode="never"
)
view.launch()
# view.launch(share=True)