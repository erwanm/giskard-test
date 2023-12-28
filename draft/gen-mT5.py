from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.optim as optim
import argparse


# GPT2
from transformers import GPT2LMHeadModel, GPT2Tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# MT5
#from transformers import MT5ForConditionalGeneration, T5Tokenizer
#model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")
#tokenizer = T5Tokenizer.from_pretrained("google/mt5-base")


# Set the device (CPU or GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


# Custom loss function (example, replace with your own)
def custom_loss(outputs, labels):
    # Your custom logic to calculate the loss
    return torch.nn.functional.cross_entropy(outputs.logits, labels)

def finetune(sentence, sentiment_model):
 # Fine-tuning data (input_ids and labels are obtained from your dataset)
    input_ids = ...
    labels = ...
    dataset = TensorDataset(input_ids, labels)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Set up optimizer
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)

    # Fine-tuning loop
    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in dataloader:
            optimizer.zero_grad()
            input_ids, labels = batch
            outputs = model(input_ids, labels=labels)
            loss = custom_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss}")

    # Save the fine-tuned model
    model.save_pretrained("fine_tuned_mt5_multilingual_generation_model")



def mymain():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_string", help="input string for generation")
    parser.add_argument("--sample_size", help="number of sentences to generate", type=int, default=100)
    parser.add_argument("--temperature", help="temperature for generation (higher=more diverse)", type=int, default=100)
    parser.add_argument("--top_k", help="top_k for generation  (higher=more diverse) ", type=int, default=100)
    args = parser.parse_args()
    print("input string is: ", args.input_string)

    # Tokenize input prompt
    input_ids = tokenizer.encode(args.input_string, return_tensors="pt").to(device)
    #print(input_ids)

    # Generate text
    for sample in range(args.sample_size):
        output = model.generate(input_ids, max_length=35, num_return_sequences=1, do_sample=True, temperature=0.95, top_k=200, pad_token_id=tokenizer.eos_token_id)

        # Decode and print the generated text
        generated_text = tokenizer.decode(output[0, input_ids.shape[-1]:], skip_special_tokens=True)
        print(f"Generated {sample}: {generated_text}")



if __name__ == '__main__':
    mymain()

