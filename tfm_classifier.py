import pandas as pd
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import torch

# Check if CUDA is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Load the IMDb dataset
dataset = load_dataset('stanfordnlp/imdb')

# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Prepare the dataset for PyTorch
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['text'])
tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
tokenized_datasets.set_format('torch')

train_dataset = tokenized_datasets['train']
test_dataset = tokenized_datasets['test']

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load the pre-trained DistilBERT model for sequence classification
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
model.to(device)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"
)

# Custom trainer class to ensure data is moved to the GPU
class CustomTrainer(Trainer):
    def training_step(self, model, inputs):
        inputs = self._prepare_inputs(inputs)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        return super().training_step(model, inputs)
    
    def evaluation_step(self, model, inputs):
        inputs = self._prepare_inputs(inputs)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        return super().evaluation_step(model, inputs)

# Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator
)

# Train the model
trainer.train()

# Evaluate the model
eval_result = trainer.evaluate()

print(f"Validation Loss: {eval_result['eval_loss']}")
print(f"Validation Accuracy: {eval_result['eval_accuracy']}")

# Inference
test_text = "I enjoyed this movie a lot!"
test_encoding = tokenizer(test_text, return_tensors='pt', truncation=True, padding=True).to(device)
model.eval()
with torch.no_grad():
    outputs = model(**test_encoding)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

print(f'Predicted class: {predicted_class}')  # 1: Positive, 0: Negative

# Save the fine-tuned model and tokenizer
model.save_pretrained('./fine-tuned-model')
tokenizer.save_pretrained('./fine-tuned-model')