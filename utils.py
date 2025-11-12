import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM

def predict_model(model, tokenizer, messages, configuration=None):
    ######################################
    ### STUB: INSERT THE CODE HERE###
    ######################################

    #if configuration provided first try to assign them to temperature and token limit if fails set basic values
    if configuration:
        temp = configuration.get("temperature", 0.1)
        max_tokens = configuration.get("max_token_limit", 2000)
    #if no configuration provided set base values to temperature and max_token_limit
    else:
        temp = 0.1
        max_tokens = 2000
    
    #use tokenizer's apply_chat_template method to reach string format that QWEN requires
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    #get model's device
    device = model.device
    #tokenize the prompt and send it to device of model
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    #retrieve the number of tokens in the prompt
    input_token_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        #call model's generate function to generate new tokens
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temp,
            do_sample=True if temp > 0.0 else False,
            pad_token_id=tokenizer.eos_token_id
        )

    #slice to get only new tokens
    generated_tokens = outputs[0][input_token_len:]
    #make model's result human readable
    model_result = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    return model_result

    

    """
    This function, `predict_model`, is designed to interact with QWEN models to generate predictions
    based on a conversation history. 

    Args:
        model: The pre-trained language model to be used for generating responses.
        tokenizer: the tokenizer corresponding to the model.
        messages: A list of dictionaries representing the conversation history,
                  where each dictionary has a "role" (e.g., "system", "user", or "assistant") 
                  and "content" (the message text).
        configuration: initially, the model used should be max_token_limit of 2000, with temperature of 0.1
        The assessment would mainly be assessed the correctness of the implementation, rather than the performance

    Returns:
        The model's response as a string.
    """


def model_evaluation(model_type, model, tokenizer, system_content, question, formatted_options, configuration=None):
    if model_type == "qwen2" or model_type == "qwen3":
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Question: {question}\n\nOptions:\n{formatted_options}"}
        ]
        model_result = predict_model(model, tokenizer, messages, configuration)
    else: 
        raise ValueError(f"Unknown model_type: {model_type}")

    #  print(f"Model result: {model_result}")
    return model_result
