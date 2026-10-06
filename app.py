import os
import gradio as gr
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

# Hugging Face client
client = InferenceClient(
    model="openai/gpt-oss-120b",
    token=HF_TOKEN
)


def generate_flashcards(topic, num_cards, difficulty):

    if not topic.strip():
        return "Please enter a study topic."

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
Make the questions appropriate for the selected difficulty.
"""

    response = client.chat_completion(
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000
    )

    return response.choices[0].message.content


# Create the web interface
with gr.Blocks(title="AI Flashcard Generator") as app:

    gr.Markdown("# 🃏 AI Flashcard Generator")
    gr.Markdown("Generate study flashcards using Hugging Face AI.")

    topic = gr.Textbox(
        label="Study Topic",
        placeholder="Example: Python Programming"
    )

    num_cards = gr.Slider(
        minimum=1,
        maximum=20,
        value=5,
        step=1,
        label="Number of Flashcards"
    )

    difficulty = gr.Dropdown(
        choices=["Easy", "Medium", "Hard"],
        value="Medium",
        label="Difficulty"
    )

    generate_button = gr.Button("Generate Flashcards")

    output = gr.Markdown(label="Generated Flashcards")

    generate_button.click(
        fn=generate_flashcards,
        inputs=[topic, num_cards, difficulty],
        outputs=output
    )


app.launch()