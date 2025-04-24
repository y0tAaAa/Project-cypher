from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('y0ta/fine_tuned_model')
tokenizer = AutoTokenizer.from_pretrained('y0ta/fine_tuned_model')
print('Model loaded successfully from Hugging Face!')
