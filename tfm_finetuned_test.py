from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch

# Load the fine-tuned model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained('./fine-tuned-model')
tokenizer = DistilBertTokenizer.from_pretrained('./fine-tuned-model')

# Move model to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Predict sentiment
test_text = "I enjoyed this movie a lot!"
test_encoding = tokenizer(test_text, return_tensors='pt', truncation=True, padding=True).to(device)
model.eval()
with torch.no_grad():
    outputs = model(**test_encoding)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

print(f'Predicted class: {predicted_class}')  # 1: Positive, 0: Negative