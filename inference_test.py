import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

local_model_path = "path/to/your/local/model/folder"

# Load the tokenizer and model from local paths
tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path)
model = AutoModelForCausalLM.from_pretrained(
  local_model_path,
  torch_dtype="auto",
)

model.cuda()

prompt = "import torch\nimport torch.nn as nn"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
tokens = model.generate(
  input,
  max_new_tokens=48,
  temperature=0.2,
  do_sample=True,
)

print(tokenizer.decode(tokens[0], skip_special_tokens=True))
