from transformers.models.auto import AutoModel, AutoTokenizer, AutoModelForSequenceClassification

print("downloading model")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
#model.save_pretrained("./model/pre_train/all-MiniLM-L6-v2")
#tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
#tokenizer.save_pretrained("./model/tokenizer/all-MiniLM-L6-v2")
print(model.parameters)
"""
print("downloading tokenizer")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base", use_fast = False)
tokenizer.save_pretrained("./model/pre_train/deberta-base")
model.save_pretrained("./model/tokenizer/deberta-base")

from transformers import DistilBertModel, DistilBertTokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
tokenizer.save_pretrained("./model/tokenizer/distilbert-base-uncased")

model = DistilBertModel.from_pretrained('distilbert-base-uncased')
model.save_pretrained("./model/pre_train/distilbert-base-uncased")
"""