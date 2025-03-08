# Phi-2 QLoRA Fine-tuning

This project demonstrates how to fine-tune the Microsoft Phi-2 model using QLoRA (Quantized Low-Rank Adaptation) technique. The implementation uses 4-bit quantization and LoRA for efficient training on consumer GPUs.

## Features

- 4-bit quantization for reduced memory usage
- LoRA adaptation for efficient fine-tuning
- Integration with Weights & Biases for experiment tracking
- Support for custom datasets
- Configurable hyperparameters

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- At least 12GB GPU memory

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/era-v3-s18-qlora-phi2.git
cd era-v3-s18-qlora-phi2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The training script (`train.py`) includes several configurable parameters:

```python
MODEL_NAME = "microsoft/phi-2"  # Base model
OUTPUT_DIR = "phi2-qlora-output"  # Output directory
LORA_R = 8  # LoRA rank
LORA_ALPHA = 32  # LoRA alpha
LORA_DROPOUT = 0.1  # LoRA dropout
LEARNING_RATE = 2e-4  # Learning rate
BATCH_SIZE = 4  # Batch size
GRADIENT_ACCUMULATION_STEPS = 4  # Gradient accumulation steps
NUM_EPOCHS = 3  # Number of training epochs
MAX_LENGTH = 512  # Maximum sequence length
```

## Training

1. Set up your Weights & Biases account and login:
```bash
wandb login
```

2. Run the training script:
```bash
python train.py
```

The script will:
- Load the Phi-2 model in 4-bit precision
- Apply LoRA adapters
- Fine-tune on the specified dataset
- Save checkpoints and metrics to W&B

## Customizing the Dataset

By default, the script uses the tiny_shakespeare dataset. To use your own dataset, modify the `prepare_dataset` function in `train.py`:

```python
def prepare_dataset(tokenizer):
    # Load your custom dataset here
    dataset = load_dataset("your_dataset")
    # ... rest of the function
```

## Model Output

The fine-tuned model will be saved in the specified `OUTPUT_DIR`. The saved model includes:
- LoRA weights
- Training configuration
- Tokenizer files

## Monitoring Training

Training progress can be monitored through:
- Console output with loss metrics
- Weights & Biases dashboard
- Saved model checkpoints

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Microsoft for the Phi-2 model
- Hugging Face for the Transformers library
- PEFT library for QLoRA implementation
