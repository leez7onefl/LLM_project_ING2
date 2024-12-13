import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

@st.cache_resource
def load_model(model_path, tokenizer_path):
    n_gpus = torch.cuda.device_count()
    max_memory = f'{6000}MB'
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        torch_dtype="auto",
        device_map="cuda:0",
        max_memory={i: max_memory for i in range(n_gpus)},
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

# Load the model and tokenizer
local_model_path = "E:/AI/projets/LLM project efrei/llm_project_M2/results/final_checkpoint_merged"
local_tokenizer_path = "E:/AI/projets/LLM project efrei/llm_project_M2/model/stable-code-3b"
model, tokenizer = load_model(local_model_path, local_tokenizer_path)
model.cuda()

def generate_response(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    tokens = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.2,
        do_sample=True,
    )
    return tokenizer.decode(tokens[0], skip_special_tokens=True)

# Streamlit UI
st.title("Stable Code 3B - Fine-tuned on Rust")
st.write("Chatbot Interface")

# Text input with default value
default_text = "Calculates and prints the size of the struct Foo in bytes"
user_input = st.text_input("Enter your prompt here:", value=default_text)

# Generate response when button clicked
if st.button("Generate Response"):
    if user_input:
        with st.spinner("Generating response..."):
            response = generate_response(user_input, model, tokenizer)
        st.write("Response:", response)
    else:
        st.write("Please enter a prompt to generate response.")
