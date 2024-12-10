from datasets import load_dataset
from functools import partial
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

import numpy as np
import random
import pandas as pd
from IPython.display import display
import evaluate
import logging

#_______________________________________________________________________

# Configurer le logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

seed = 69
set_seed(seed)
n_gpus = torch.cuda.device_count()
print(torch.__version__)
print(torch.cuda.is_available())
print("NOMBRE DE GPU : ", n_gpus)

accuracy_metric = evaluate.load("accuracy")

#_______________________________________________________________________

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

#_______________________________________________________________________

def load_model(model_path):
    n_gpus = torch.cuda.device_count()
    max_memory = f'{32000}MB'
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cuda:0",
        max_memory={i: max_memory for i in range(n_gpus)},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

#_______________________________________________________________________

dataset = load_dataset('json', data_files='E:/AI/projets/LLM project efrei/llm_project_M2/data/data_code.json', split='train')

print(f'Number of prompts: {len(dataset)}')
print(f'Column names are: {dataset.column_names}')

nb_samples = 3
random_indices = random.sample(range(len(dataset)), nb_samples)
samples = []
for idx in random_indices:
    sample = dataset[idx]
    sample_data = {
        'instruction': sample['instruction'],
        'input': sample['input'],
        'output': sample['output']
    }
    samples.append(sample_data)
df = pd.DataFrame(samples)
display(df)

#_______________________________________________________________________

def create_prompt_formats(sample):
    INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    INSTRUCTION_KEY = "### Instruction:"
    INPUT_KEY = "### Input:"
    RESPONSE_KEY = "### Output:"
    END_KEY = "### End"
    
    blurb = f"{INTRO_BLURB}"
    instruction = f"{INSTRUCTION_KEY}\n{sample['instruction']}"
    input_context = f"{INPUT_KEY}\n{sample['input']}" if sample["input"] else None
    response = f"{RESPONSE_KEY}\n{sample['output']}"
    end = f"{END_KEY}"
    parts = [part for part in [blurb, instruction, input_context, response, end] if part]
    formatted_prompt = "\n\n".join(parts)
    sample["text"] = formatted_prompt
    return sample

print(create_prompt_formats(dataset[0])["text"])

#_______________________________________________________________________

def get_max_length(model):
    conf = model.config
    max_length = getattr(conf, 'n_positions', None) or getattr(conf, 'max_position_embeddings', None) or getattr(conf, 'seq_length', None) or 1024
    return max_length

#_______________________________________________________________________

def preprocess_batch(batch, tokenizer, max_length):
    return tokenizer(batch["text"], max_length=max_length, truncation=True)

#_______________________________________________________________________

def preprocess_dataset(tokenizer, max_length, seed, dataset):
    print("Preprocessing dataset...")
    dataset = dataset.map(create_prompt_formats)
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(_preprocessing_function, batched=True, remove_columns=["instruction", "input", "output", "text"])
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)
    dataset = dataset.shuffle(seed=seed)
    return dataset

#_______________________________________________________________________

def create_peft_config(modules):
    return LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules=modules,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

#_______________________________________________________________________

def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

#_______________________________________________________________________

def print_trainable_parameters(model):
    trainable_params, all_param = 0, 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}")

#_______________________________________________________________________

model_path = "E:/AI/projets/LLM project efrei/llm_project_M2/model/stable-code-3b"
model, tokenizer = load_model(model_path)

max_length = get_max_length(model)

# Split the dataset into training and evaluation sets
split_ratio = 0.3
splits = dataset.train_test_split(test_size=split_ratio, seed=seed)
train_dataset = splits['train']
eval_dataset = splits['test']

# Preprocess datasets
train_dataset = preprocess_dataset(tokenizer, max_length, seed, train_dataset)
eval_dataset = preprocess_dataset(tokenizer, max_length, seed, eval_dataset)

#_______________________________________________________________________

# Setup directories upfront
output_dir = "E:/AI/projets/LLM project efrei/llm_project_M2/results/final_checkpoint"
output_merged_dir = "E:/AI/projets/LLM project efrei/llm_project_M2/results/final_checkpoint_merged"

# Create directories if not exists
for dir_path in [output_dir, output_merged_dir]:
    os.makedirs(dir_path, exist_ok=True)

#_______________________________________________________________________

def train(model, tokenizer, train_dataset, eval_dataset, output_dir):
    logger.info("Starting the training process.")

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    logger.info("Gradient checkpointing enabled.")

    # Prepare model for k-bit training and locate linear modules
    model = prepare_model_for_kbit_training(model)
    modules = find_all_linear_names(model)
    logger.info(f"Model prepared for k-bit training. Located linear modules: {modules}")

    # Obtain PEFT model
    peft_config = create_peft_config(modules)
    model = get_peft_model(model, peft_config)
    logger.info("PEFT model obtained.")

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=0,
            max_steps=1,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            output_dir="outputs",
            optim="paged_adamw_8bit",
            eval_strategy="no",
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        compute_metrics=compute_metrics,
    )
    logger.info("Trainer initialized.")

    model.config.use_cache = False

    # Training
    train_result = trainer.train()
    logger.info(f"Training completed. Metrics: {train_result.metrics}")

    # Evaluate
    eval_result = trainer.evaluate()
    logger.info(f"Evaluation completed. Results: {eval_result}")

    # Save trained model
    trainer.model.save_pretrained(output_dir)

    # Clear resources
    del model
    del trainer
    torch.cuda.empty_cache()
    logger.info("Training process completed and resources cleaned up.")

# Run training process
train(model, tokenizer, train_dataset, eval_dataset, output_dir)

# Merge weights if needed
model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="cuda:0", torch_dtype=torch.bfloat16)
model = model.merge_and_unload()
model.save_pretrained(output_merged_dir, safe_serialization=True)
tokenizer.save_pretrained(output_merged_dir)