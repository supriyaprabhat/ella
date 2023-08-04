from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import pandas as pd

df = pd.read_csv("ML/data-preprocesing/Python/Data.csv")



tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

tokenizer.mask_token = None
tokenizer.pad_token = None
model.config.pad_token_id = None
model.config.mask_token_id = None

dataset = []
for index, row in df.iterrows():
    user_query = row["user_query"]
    bot_response = row["bot_response"]
    dataset.append({"user": user_query, "response": bot_response})

tokenized_dataset = tokenizer([conv["user"] for conv in dataset], truncation=True, padding=True)

input_ids = torch.tensor(tokenized_dataset["input_ids"])
attention_mask = torch.tensor(tokenized_dataset["attention_mask"])
target_ids = input_ids.clone()

def chatbot_response(user_input, max_length=100):
    # Encode user input to tensor
    input_ids = tokenizer.encode(user_input, return_tensors="pt")

    # Generate a response using the model
    with torch.no_grad():
        response_ids = model.generate(input_ids, max_length=max_length, num_return_sequences=1)

    # Decode the response tensor to text
    response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)

    return response_text

def main():
    print("Chatbot: Hi there! I'm your AI chatbot. Ask me anything or say 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break

        response = chatbot_response(user_input)
        print("Chatbot:", response)

if __name__ == "__main__":
    main()
