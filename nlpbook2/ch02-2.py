import warnings
from urllib3.exceptions import NotOpenSSLWarning
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel,AutoTokenizer

path= '/models/gpt2/'
# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained(path)
# tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
# Encode a text inputs
text = "With great power comes great "
indexed_tokens = tokenizer.encode(text)

# Convert indexed tokens in a PyTorch tensor
tokens_tensor = torch.tensor([indexed_tokens])

print(tokens_tensor)

# Load pre-trained model (weights)
model = GPT2LMHeadModel.from_pretrained(path)

# Set the model in evaluation mode to deactivate the DropOut modules
# This is IMPORTANT to have reproducible results during evaluation!
model.eval()

# Predict all tokens
with torch.no_grad():
    outputs = model(tokens_tensor)
    predictions = outputs[0]

# get the predicted next sub-word
predicted_index = torch.argmax(predictions[0, -1, :]).item()
predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
print(predicted_text)

# model = GPT2LMHeadModel.from_pretrained(path)
#
# model.eval()
#
# with torch.no_grad():
#     outputs = model(tokens_tensor)
#     predictions = outputs[0]
#
# outputs[0].shape
#
# with torch.no_grad():
#     outputs = model(tokens_tensor)
#     predictions = outputs[0]
#
# predicted_index = torch.argmax(predictions[0, -1, :]).item()
#
# predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
# print(predicted_text)