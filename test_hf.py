import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(model="google/gemma-2-2b-it",
    token= HF_TOKEN
)

response = client.text_generation(
    "Explain artificial intelligence in simple terms.",
    max_new_tokens=100
)

print(response)