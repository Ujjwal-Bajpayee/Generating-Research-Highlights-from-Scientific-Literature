#pegasus experimentation which has achieved max score
import pandas as pd
from datasets import Dataset
from transformers import PegasusTokenizer

file_path = "data/MixSub-SciHigh_train_FIRE.csv"

def get_tokenized_dataset(file_path, tokenizer_name="google/pegasus-large"):
    df = pd.read_csv(file_path)
    dataset = Dataset.from_pandas(df)
    
    tokenizer = PegasusTokenizer.from_pretrained(tokenizer_name)

    def preprocess_function(examples):
        model_inputs = tokenizer(examples["Abstract"], max_length=512, truncation=True, padding="max_length")
        
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["Highlights"], max_length=100, truncation=True, padding="max_length")

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(preprocess_function, batched=True), tokenizer