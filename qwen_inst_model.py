from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
pipe = pipeline("text-generation", model=model_name)

text = 'Hello,'
inputs = tokenizer(
    text,
    return_tensors="pt",
)
print(inputs.input_ids.shape)

outputs = model.generate(**inputs,
                         max_new_tokens=128,
                         do_sample=False,
                         top_k=None,
                         top_p=None,
                         temperature=None
                         )

print(tokenizer.batch_decode(outputs))

messages = [
    {"role": "user", "content": "Who are you?"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)
print(inputs.input_ids.shape)

outputs = model.generate(**inputs,
    max_new_tokens=128,
    do_sample=False,
    top_k=None,
    top_p= None,
    temperature=None
)
print(tokenizer.batch_decode(outputs))
messages = [
    {"role": "user", "content": "Who are you"},
]
print(pipe(messages))

text = 'Hello,'
print(pipe(text, max_new_tokens=128))
