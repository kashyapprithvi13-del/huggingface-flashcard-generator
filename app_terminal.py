import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(
    model="openai/gpt-oss-120b",
    token=HF_TOKEN
)

topic = input("Enter a study topic: ")
num_cards = input("How many flashcards do you want? ")
difficulty = input("Choose difficulty (Easy/Medium/Hard): ")

prompt = f"""
You are an educational flashcard generator.

Create {num_cards} flashcards about:

Topic: {topic}
Difficulty: {difficulty}
For each flashcard provide:

Question:
Answer:

Keep the questions clear and suitable for a student.
Keep the answers short and accurate.
"""

response = client.chat_completion(
    messages=[
        {"role": "user", "content": prompt}
    ],
    max_tokens=500
)

response = response.choices[0].message.content

print("\nGenerated Flashcards:\n")
print(response)