# Documentation for Fine-Tuning Scripts

## Part 1: Project Description

The primary goal of this project is to refine the `stable-code-3b` model, a transformer language model with 2.7 billion parameters, specifically engineered for code comprehension and generation across diverse programming languages.

A significant challenge within this project is adapting the model to process new data efficiently while maintaining optimal performance and accuracy. The utilized dataset consists of large-scale, open-source textual and code data, which has been meticulously filtered and tailored for this task. This initiative aims to enhance code generation capabilities by methodically adjusting the model's weights using specific fine-tuning scripts. 

A particularly noteworthy challenge was to confine the fine-tuning process to the capabilities of a GeForce RTX 3080 graphics card, equipped with 10GB of VRAM, thus enabling a completely local setup without reliance on cloud-based infrastructure. Adapting such a substantial model within limited GPU memory necessitated innovative strategies and optimizations.

## Part 2: Function Explanation

```python
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
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)

import numpy as np
import random
import pandas as pd
from IPython.display import display
import evaluate
import logging

import warnings
warnings.filterwarnings("ignore")

# Configurer le logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

seed = 69
set_seed(seed)
n_gpus = torch.cuda.device_count()
print(torch.cuda.get_device_name(0))

print(torch.__version__)
print(torch.cuda.is_available())
print("NOMBRE DE GPU : ", n_gpus)

accuracy_metric = evaluate.load("accuracy")
```

- Importation of libraries and settings


### `compute_metrics(eval_pred)`

This function calculates the accuracy metric to evaluate model performance:

```python
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)
```

- **Line 1-2**: The function inputs evaluate predictions consisting of logits (raw model outputs) and labels (ground truths).
- **Line 3**: Employs `numpy` to identify the class with the highest score from the logits for each sample.
- **Line 4**: Computes and returns accuracy using the `accuracy_metric`.

### `load_model(model_path)`

This function loads a pre-trained model and tokenizer from a specified path, optimized for GPU efficiency:

```python
def load_model(model_path):
    n_gpus = torch.cuda.device_count()
    max_memory = f'{10000}MB'
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        torch_dtype="auto",
        device_map="cuda:0",
        max_memory={i: max_memory for i in range(n_gpus)},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
```

- **Line 2**: Counts the available GPUs, enabling resource allocation.
- **Line 3**: Establishes the memory limit per GPU.
- **Line 4-9**: Loads the causal language model using 8-bit precision to economize memory usage
- **Line 10**: Initializes the tokenizer from the specified model path.
- **Line 12**: Uses the `eos_token` as the `pad_token` for consistent outputs.
- **Line 13**: Returns the model and tokenizer for subsequent use.

```python
dataset = load_dataset('json', data_files='E:/AI/projets/LLM project efrei/llm_project_M2/data/data_code.json')

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
```

- Load and display informations about dataset

### `create_prompt_formats(sample)`

This function organizes sample data into structured prompt formats:

```python
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
```

- **Line 2-6**: Sets constant strings defining sections of the prompt.
- **Line 8-12**: Constructs parts of the formatted text attached to each section.
- **Line 13**: Assembles all non-null sections into a single formatted string.
- **Line 15**: Adds the formatted string to the sample dictionary under `text`.
- **Line 16**: Returns the sample with the formatted text.

### `get_max_length(model)`

This function retrieves the maximum sequence length the model supports:

```python
def get_max_length(model):
    conf = model.config
    max_length = getattr(conf, 'n_positions', None) or getattr(conf, 'max_position_embeddings', None) or getattr(conf, 'seq_length', None) or 1024
    print(max_length)
    return max_length
```

- **Line 2**: Accesses the model's configuration.
- **Line 3**: Retrieves the maximum sequence length, defaulting to 1024 if not specified.
- **Line 4**: Prints the identified maximum length for verification.
- **Line 5**: Returns the maximum sequence length.

### `preprocess_batch(batch, tokenizer, max_length)`

This function tokenizes a batch of data to fit the maximum model input length:

```python
def preprocess_batch(batch, tokenizer, max_length):
    return tokenizer(batch["text"], max_length=max_length, truncation=True)
```

- **Line 2**: Tokenizes text within the given batch, enforcing length restrictions and truncating as needed.

### `preprocess_dataset(tokenizer, max_length, seed, dataset)`

This function conducts formatting and tokenization of the whole dataset for training readiness:

```python
def preprocess_dataset(tokenizer, max_length, seed, dataset):
    print("Preprocessing dataset...")
    dataset = dataset.map(create_prompt_formats)
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(_preprocessing_function, batched=True, remove_columns=["instruction", "input", "output", "text"])
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)
    dataset = dataset.shuffle(seed=seed)
    return dataset
```

- **Line 2**: Outputs a preprocessing initialization message.
- **Line 3**: Applies `create_prompt_formats` to format each dataset entry.
- **Line 4**: Constructs a partial function for batch-level preprocessing.
- **Line 5**: Applies preprocessing, removing specified columns.
- **Line 6**: Filters samples that exceed the maximum tokenized length.
- **Line 7**: Shuffles the dataset using the provided seed for consistency.
- **Line 8**: Returns the preprocessed dataset.

### `create_peft_config(modules)`

This function configures the model for Parameter-Efficient Fine-Tuning (PEFT) using LoRA:

```python
def create_peft_config(modules):
    return LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
```

- **Line 2-8**: Instantiates `LoraConfig`, detailing hyperparameters like rank, alpha, targeted model layers, dropout, bias application, and task specification, optimizing fine-tuning efficiency.

### `find_all_linear_names(model)`

This function identifies all linear layers within the model suitable for LoRA adaptation:

```python
def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)
```

- **Line 2**: Initializes a set for capturing names of linear modules.
- **Line 3-6**: Iterates over model modules, identifying applicable linear layers.
- **Line 7**: Excludes 'lm_head' as it is not to be modified by LoRA.
- **Line 8**: Converts names to a list and returns, designating layers for adaptation.

### `print_trainable_parameters(model)`

This function calculates and logs the total and trainable parameters in the model:

```python
def print_trainable_parameters(model):
    trainable_params, all_param = 0, 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}")
```

- **Line 2**: Initializes counters for parameter tracking.
- **Line 3-7**: Totals all and only trainable parameters flagged `requires_grad`.
- **Line 8**: Summarizes total, trainable counts, and proportion as a percentage.

### `train(model, tokenizer, train_dataset, eval_dataset, output_dir)`

This function orchestrates data preprocessing, model configuration, training, and evaluation:

```python
def train(model, tokenizer, train_dataset, eval_dataset, output_dir):
    logger.info("Starting the training process.")

    # Enable gradient checkpointing
    # model.gradient_checkpointing_enable()
    # logger.info("Gradient checkpointing enabled.")

    # Prepare model for k-bit training and locate linear modules
    model = prepare_model_for_kbit_training(model)
    
    # Freeze all layers except the last one
    for name, param in model.named_parameters():
        if "lm_head" not in name:  # Change "lm_head" according to the name of the final layer in your model
            param.requires_grad = False
        else:
            param.requires_grad = True

    # Output the trainable parameters (this should now be only the last layer)
    print_trainable_parameters(model)

    # Obtain PEFT model
    modules = find_all_linear_names(model)
    logger.info(f"Model prepared for k-bit training. Located linear modules: {modules}")
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
            learning_rate=1e-5,
            logging_steps=10,
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
    logger.info(f"BEGINNING TRAINING")
    train_result = trainer.train()
    logger.info(f"Training completed. Metrics: {train_result.metrics}")

    # Evaluate
    logger.info(f"BEGINNING EVALUATION")
    eval_result = trainer.evaluate()
    logger.info(f"Evaluation completed. Results: {eval_result}")

    # Save trained model
    trainer.model.save_pretrained(output_dir)

    # Clear resources
    del model
    del trainer
    torch.cuda.empty_cache()
    logger.info("Training process completed and resources cleaned up.")
```

- **Line 2**: Logs initiation of training.
- **Line 8**: Prepares model for k-bit precision, enhancing computational efficiency.
- **Line 11-15**: Limits trainable parameters to terminal layers, optimizing updates.
- **Line 18**: Logs information about trainable parameters.
- **Line 21**: Collects linear module names for adaptation.
- **Line 22**: Logs discovery of linear modules.
- **Line 23**: Establishes LoRA adaptation configuration.
- **Line 24**: Applies the adaptation configuration to the model.
- **Line 25**: Confirms creation of the PEFT model.
- **Line 28-41**: Sets up the `Trainer` with relevant training settings.
- **Line 42**: Logs trainer readiness.
- **Line 44**: Disables model cache usage, minimizing memory requirements.
- **Line 47-49**: Conducts model training and logs completion metrics.
- **Line 52-54**: Executes and logs evaluation results.
- **Line 57**: Saves the model state post-training.
- **Line 60-62**: Releases resources and clears GPU memory, ensuring readiness for future tasks.

```python
# Run training process
train(model, tokenizer, train_dataset, eval_dataset, output_dir)

# Merge weights
model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="cuda:0")
model = model.merge_and_unload()
model.save_pretrained(output_merged_dir, safe_serialization=True)
tokenizer.save_pretrained(output_merged_dir)
```
- Run the training, merge and save weights

## Part 3: Processes and Choices Explanation

A series of strategic decisions were undertaken to address the model's size and ensure successful fine-tuning within constrained computational environments:

- **LoRA Technique**: The Low-Rank Adaptation (LoRA) method focused updates on select model components, reducing parameter numbers and computational demands. This facilitated efficient memory use on available hardware during fine-tuning.

- **Quantization and 4bit Precision**: The deployment of 8bit precision markedly minimized VRAM consumption, enabling the large model to function on a GeForce RTX 3080, which possesses restricted VRAM capacity.

- **Freezing Layers Technique**: By immobilizing all but the final model layers, we significantly decreased trainable parameter counts, aligning resources with the PEFT approach.

- **PYTORCH_CUDA_ALLOC_CONF**: The configuration of this environment variable (`max_split_size_mb:128`) strategically handled memory fragmentation in CUDA allocations, mitigating potential memory limitation issues.

- **Gradient Checkpointing**: Although not actively utilized within the current framework, gradient checkpointing presents a potential avenue for reducing memory overhead by refraining from caching intermediate activations, instead recalculating them during the backpropagation phase.

Faced with the primary constraint of localizing the fine-tuning process of a large model within the confines of a GeForce RTX 3080 (10GB VRAM), these techniques collectively facilitated substantial advancements. By strategically managing computational and memory demands, the project achieved comprehensive local model training, thereby broadening access to state-of-the-art AI technologies without cloud dependencies. Such advancements demonstrate the feasibility of employing advanced models on limited resource setups, opening avenues for widespread application in diverse settings.
