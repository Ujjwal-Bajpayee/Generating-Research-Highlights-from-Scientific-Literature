import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

def generate_highlights(text, model_path="./models/pegasus_final"):
    tokenizer = PegasusTokenizer.from_pretrained(model_path)
    model = PegasusForConditionalGeneration.from_pretrained(model_path).to("cuda")

    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to("cuda")

    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,               
        max_length=100,            
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)