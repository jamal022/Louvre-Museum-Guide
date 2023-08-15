import os
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load GPT-2 model and tokenizer
MODEL_PATH = "F:\GuideProject\model"  # Update this with your actual model path
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def generate_text(sequence, max_length):
    ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=max_length,
        pad_token_id=model.config.eos_token_id,
        top_k=50,
        top_p=0.95,
    )
    generated_text = tokenizer.decode(final_outputs[0], skip_special_tokens=True)

    # Find the first occurrence of [Guide] and [Visitor] in the generated text
    guide_start_index = generated_text.find('[Guide]')
    visitor_start_index = generated_text.find('[Visitor]')

    # Extract the generated text between [Guide] and [Visitor]
    if guide_start_index != -1 and visitor_start_index != -1:
        generated_text = generated_text[guide_start_index + 8:visitor_start_index].strip()
    elif guide_start_index != -1:
        generated_text = generated_text[guide_start_index + 8:].strip()

    return generated_text


@app.route("/", methods=["GET"])
def index():
    return render_template('index.html')


@app.route("/generate_text", methods=["POST"])
def handle_generate_text():
    data = request.get_json()
    sequence = data.get("sequence", "")
    max_length = data.get("max_length", 100)  # You can adjust this to control the response length

    generated_text = generate_text(sequence, max_length)

    return jsonify({"generated_text": generated_text})


if __name__ == "__main__":
    app.run(debug=True)
