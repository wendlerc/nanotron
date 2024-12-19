from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description="Add a new special token to a model.")
parser.add_argument("--model_name", type=str, required=True, help="The name of the model to update.")
parser.add_argument("--save_dir", type=str, required=True, help="The directory to save the updated model and tokenizer.")
args = parser.parse_args()

# Step 1: Load the tokenizer and model
model_name = args.model_name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Step 2: Add a new special token
new_special_token = "<PAUSE>"
if new_special_token not in tokenizer.special_tokens_map:
    tokenizer.add_special_tokens({"additional_special_tokens": [new_special_token]})
    model.resize_token_embeddings(len(tokenizer))  # Resize embeddings for new token

# Step 3: Save the updated tokenizer and model
save_directory = args.save_dir
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

print(f"Updated model and tokenizer saved to: {save_directory}")
