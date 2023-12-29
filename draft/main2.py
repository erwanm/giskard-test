import torch
from torch.nn import CrossEntropyLoss
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Define custom loss function
def custom_loss(logits, labels):
    # Your custom loss calculation here
    return CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))

# Function to generate synthetic training data using the fine-tuned model
def generate_synthetic_data(model, tokenizer, num_samples=1000, max_length=50):
    generated_data = []
    for _ in range(num_samples):
        input_text = tokenizer.encode("Your prompt here", return_tensors="pt")
        output = model.generate(input_text, max_length=max_length, num_return_sequences=1)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_data.append(generated_text)
    return generated_data

# Set your synthetic training data file path
synthetic_data_path = "./synthetic_data.txt"

# Load pre-trained GPT-2 model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token=tokenizer.eos_token

# Generate synthetic training data
synthetic_data = generate_synthetic_data(model, tokenizer, num_samples=50)
with open(synthetic_data_path, 'w', encoding='utf-8') as file:
    file.write('\n'.join(synthetic_data))

# Tokenize the synthetic training data
synthetic_train_data = TextDataset(
    tokenizer=tokenizer,
    file_path=synthetic_data_path,
    block_size=128  # Adjust block_size according to your data and GPU memory
)

# Create data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Not using MLM for sentence generation
)

# Set training arguments
training_args = TrainingArguments(
    output_dir="./fine-tuned-gpt2",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

# Create Trainer with custom loss function
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=synthetic_train_data,  # Use synthetic training data
    compute_metrics=None,  # Set to None to use custom loss function
)

# Override the default training step to use custom loss
def compute_loss(model, inputs):
    outputs = model(**inputs)
    logits = outputs.logits
    labels = inputs["labels"]
    loss = custom_loss(logits, labels)
    return loss

trainer.compute_loss = compute_loss

# Fine-tune the GPT-2 model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine-tuned-gpt2")
tokenizer.save_pretrained("./fine-tuned-gpt2")
