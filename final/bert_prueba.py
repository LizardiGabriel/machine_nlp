#pip install datasets
#pip install transformers[torch]

import torch
from datasets import list_datasets
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
from transformers import BertTokenizer, TrainingArguments, BertForSequenceClassification, Trainer


emotion_corpus = load_dataset("emotion")

emotion_corpus_train = emotion_corpus['train']
emotion_corpus_validation = emotion_corpus['validation']
emotion_corpus_test = emotion_corpus['test']
emotion_corpus_train.set_format(type="pandas")
emotion_corpus_validation.set_format(type="pandas")
emotion_corpus_test.set_format(type="pandas")
df_emotion_corpus_train = emotion_corpus_train[:]
df_emotion_corpus_validation = emotion_corpus_validation[:]
df_emotion_corpus_test = emotion_corpus_test[:]

print (df_emotion_corpus_validation)

df_emotion_corpus_validation["label_name"] = df_emotion_corpus_validation["label"].apply(lambda x: emotion_corpus_validation.features["label"].int2str(x))
print(df_emotion_corpus_validation[["text", "label_name"]])
print (df_emotion_corpus_validation["label_name"])
print (type(df_emotion_corpus_validation["label_name"]))
label_name = df_emotion_corpus_validation["label_name"].unique()
print (label_name)

#https://huggingface.co/bert-base-uncased
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

#Training using pretrained model weights
model_ckpt = "bert-base-uncased"
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = (BertForSequenceClassification.from_pretrained(model_ckpt, num_labels=6))



from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


class Dataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels=None):
        self.encodings = tokenizer(texts, padding=True, truncation=True, max_length=512)
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.encodings["input_ids"])


train_dataset = Dataset(emotion_corpus["train"]["text"].tolist(), emotion_corpus["train"]["label"].tolist())
validation_dataset = Dataset(emotion_corpus["validation"]["text"].tolist(), emotion_corpus["validation"]["label"].tolist())
test_dataset = Dataset(emotion_corpus["test"]["text"].tolist(), emotion_corpus["test"]["label"].tolist())


print (train_dataset.encodings["input_ids"][0])
tokens = tokenizer.convert_ids_to_tokens(train_dataset.encodings["input_ids"][0])
print (tokens)
print(tokenizer.convert_tokens_to_string(tokens))
print (train_dataset.labels[0])


training_args = TrainingArguments(
    output_dir="output",
    evaluation_strategy="steps",
    eval_steps=100,
    num_train_epochs=2,
    seed=0,
    load_best_model_at_end=True,
    fp16=True)


trainer = Trainer(
    model=model,
    args = training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer
)
trainer.train()


predictions = trainer.predict(test_dataset)
print (predictions.metrics)



from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def plot_confusion_matrix(y_preds, y_true, labels):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.show()


import numpy as np

print (predictions.predictions)
y_preds = np.argmax(predictions.predictions, axis=-1)
print (y_preds)
y_test = emotion_corpus["test"]["label"].tolist()
print (y_test)
plot_confusion_matrix(y_preds, y_test, label_name)


from sklearn.metrics import classification_report
print(classification_report(y_test, y_preds, target_names=label_name))


