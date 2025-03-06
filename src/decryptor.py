# src/decryptor.py
from transformers import AutoModelForCausalLM, AutoTokenizer

class Decryptor:
    def __init__(self, model_path: str, cipher_type: str = "Caesar"):
        # Load the pretrained model and tokenizer from Hugging Face
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.cipher_type = cipher_type

    def decrypt(self, ciphertext: str, max_length: int = 150) -> str:
        # Construct prompt to feed the model
        prompt = f"Cipher: {self.cipher_type}\nCiphertext: {ciphertext}\nPlaintext:"
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")

        # Generate output from the model
        outputs = self.model.generate(
            inputs, 
            max_length=inputs.shape[1] + max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            num_return_sequences=1
        )

        # Decode the generated text
        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract plaintext from output
        plaintext = decoded_output.split("Plaintext:")[-1].strip()
        return plaintext

if __name__ == "__main__":
    # Example usage with GPT-2 model
    model_path = "gpt2"
    decryptor = Decryptor(model_path, cipher_type="Caesar")

    sample_ciphertext = "KHOOR ZRUOG"  # "HELLO WORLD" with Caesar cipher shift 3
    decrypted_text = decryptor.decrypt(sample_ciphertext)

    print(f"Ciphertext: {sample_ciphertext}")
    print(f"Decrypted Text: {decrypted_text}")
