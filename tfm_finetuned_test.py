import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

# Load the fine-tuned model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained('./fine-tuned-model')
tokenizer = DistilBertTokenizer.from_pretrained('./fine-tuned-model')

# Example prediction
test_text = "I enjoyed this movie a lot!"
test_encoding = tokenizer(test_text, return_tensors='pt', truncation=True, padding=True)
model.eval()
with torch.no_grad():
    outputs = model(**test_encoding)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

print(f'Predicted class: {predicted_class}')  # 1: Positive, 0: Negative