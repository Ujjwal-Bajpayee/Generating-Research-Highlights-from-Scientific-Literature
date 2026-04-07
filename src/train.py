#we stopped because it was showing around 65 hrs for training 

from transformers import PegasusForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from preprocess import get_tokenized_dataset

def train_model(train_path, val_path):
    tokenized_train, tokenizer = get_tokenized_dataset(train_path)
    tokenized_val, _ = get_tokenized_dataset(val_path)
    
    model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-large")

    training_args = TrainingArguments(
        output_dir="./models/pegasus_sota",
        num_train_epochs=10,             
        per_device_train_batch_size=2,    
        learning_rate=2e-5,               
        optim="adamw_torch",         
        eval_strategy="epoch",        
        save_strategy="epoch",
        logging_steps=100,
        fp16=True,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )

    trainer.train()
    model.save_pretrained("./models/pegasus_final")
    tokenizer.save_pretrained("./models/pegasus_final")

if __name__ == "__main__":
    train_model("data/MixSub-SciHigh_train_FIRE.csv", "data/MixSub-SciHigh_val_FIRE.csv")