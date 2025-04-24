from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("fine_tuned_model_new")
tokenizer = AutoTokenizer.from_pretrained("fine_tuned_model_new")
model.push_to_hub("y0ta/fine_tuned_model")
tokenizer.push_to_hub("y0ta/fine_tuned_model")
print("Model uploaded successfully!")
