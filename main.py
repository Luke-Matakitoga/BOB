from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model and tokenizer from Hugging Face
model_name = "microsoft/DialoGPT-medium"  # You can replace this with a different conversational model

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize the chat history
chat_history_ids = None

print("Model loaded. Start chatting!")

# Chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Bot: Goodbye!")
        break

    # Encode the user input
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")

    # Append the user input to the chat history (if exists)
    chat_history_ids = (
        torch.cat([chat_history_ids, new_input_ids], dim=-1)
        if chat_history_ids is not None
        else new_input_ids
    )

    # Generate a response
    response_ids = model.generate(
        chat_history_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        top_p=0.92,
        top_k=50
    )

    # Decode the response
    response = tokenizer.decode(response_ids[:, chat_history_ids.shape[-1]:][0], skip_special_tokens=True)
    print(f"Bot: {response}")
