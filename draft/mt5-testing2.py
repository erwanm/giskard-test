from transformers import MT5ForConditionalGeneration, T5Tokenizer

# Load mT5 model and tokenizer
model = MT5ForConditionalGeneration.from_pretrained("google/mt5-large")
tokenizer = T5Tokenizer.from_pretrained("google/mt5-large",legacy=True)

# Example prompt
prompt = "Translate English to French: 'Hello, how are you?'"

# Tokenize and generate output
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output_ids = model.generate(input_ids, max_new_tokens=50)

# Decode and print the output
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output_text)
