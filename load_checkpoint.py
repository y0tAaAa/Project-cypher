from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained('results/checkpoint-75000')
tokenizer = AutoTokenizer.from_pretrained('fine_tuned_model')
model.save_pretrained('fine_tuned_model_new', safe_serialization=False)
tokenizer.save_pretrained('fine_tuned_model_new')
print('Model loaded and saved successfully!')