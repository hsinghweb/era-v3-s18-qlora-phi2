import os
import torch
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model
)
from datasets import load_dataset
import wandb
from tqdm import tqdm

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Extract configurations
MODEL_NAME = config['model']['name']
OUTPUT_DIR = config['model']['output_dir']
LORA_R = config['lora']['rank']
LORA_ALPHA = config['lora']['alpha']
LORA_DROPOUT = config['lora']['dropout']
LEARNING_RATE = config['training']['learning_rate']
BATCH_SIZE = config['training']['batch_size']
GRADIENT_ACCUMULATION_STEPS = config['training']['gradient_accumulation_steps']
NUM_EPOCHS = config['training']['num_epochs']
MAX_LENGTH = config['training']['max_length']

# Initialize wandb
wandb.init(project="phi2-qlora-finetuning")

def load_model_and_tokenizer():
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model in 4-bit precision
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "dense"]
    )
    
    # Get PEFT model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def prepare_dataset(tokenizer):
    # Load dataset based on configuration
    dataset = load_dataset(config['dataset']['name'])
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=config['dataset']['truncation'],
            max_length=MAX_LENGTH,
            padding=config['dataset']['padding']
        )
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    return tokenized_dataset

def main():
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()
    
    print("Preparing dataset...")
    dataset = prepare_dataset(tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        report_to="wandb",
        fp16=config['training']['fp16'],
        remove_unused_columns=False
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model()
    
    # End wandb run
    wandb.finish()

if __name__ == "__main__":
    main()