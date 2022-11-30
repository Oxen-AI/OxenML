
from datasets import load_dataset
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import AutoModel
from umap import UMAP
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import torch
import argparse

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_name).to(device)

def tokenize(batch):
    global tokenizer
    return tokenizer(batch['text'], padding=True, truncation=True)

def extract_hidden_states(batch):
    global tokenizer, model
    inputs = {k:v.to(device) for k,v in batch.items() if k in tokenizer.model_input_names}
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
    return {"hidden_state": last_hidden_state[:,0].cpu().numpy()}

def plot_confusion_matrix(y_preds, y_true, labels):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6,6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized Confusion Matrix")
    plt.show()

def train(raw_args):
    global device, tokenizer

    parser = argparse.ArgumentParser(
        description="Command line tool to train a text classification model off of an input dataset"
    )

    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        required=True,
        help="The input file of data",
    )
    args = parser.parse_args(raw_args)
    
    # dataset = load_dataset("csv", data_files=args.input_file, sep='\t', names=["text", "label"])
    
    df = pd.read_csv(args.input_file, on_bad_lines='warn', delimiter='\t')    
    dataset = DatasetDict()
 
    split_on = int(df.shape[0] * 0.8)
    dataset['train'] = Dataset.from_pandas(df[0:split_on])
    dataset['test'] = Dataset.from_pandas(df[split_on:])
    
    # Convert strings to labels
    dataset = dataset.class_encode_column("label")
    print(dataset['train'][:5])

    ## Example of tokenizing one example
    # example = dataset['train'][0]
    # print(f"Got example: {example}")
    
    # encoded_text = tokenizer(example['text'])
    # print(f"Encoded: {encoded_text}")
    
    # tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
    # print(f"Tokens: {tokens}")
    
    
    # Tokenize the full dataset
    print("Tokenizing dataset....")
    dataset_encoded = dataset.map(tokenize, batched=True, batch_size=None)
    dataset_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    print("Computing hidden states....")
    dataset_hidden = dataset_encoded.map(extract_hidden_states, batched=True)

    # print(dataset_hidden['train'].column_names)
    # example = dataset_hidden['train'][0]
    # print(f"Got example: {example}")
    
    X_train = np.array(dataset_hidden['train']['hidden_state'])
    X_test = np.array(dataset_hidden['test']['hidden_state'])

    y_train = np.array(dataset_hidden['train']['label'])
    y_test = np.array(dataset_hidden['test']['label'])

    print(f"X_train.shape: {X_train.shape}")
    print(f"X_test.shape: {X_test.shape}")

    labels = dataset["train"].features["label"].names
    print(f"Labels: {labels}")


    ## Helpful to see if it is going to be easy to split the categories
    # print("UMAPing...")
    # X_scaled = MinMaxScaler().fit_transform(X_train)
    # mapper = UMAP(n_components=2, metric="cosine").fit(X_scaled)
    
    # df_emb = pd.DataFrame(mapper.embedding_, columns=["X", "Y"])
    # df_emb["label"] = y_train
    # print(df_emb.head())
    # fig, axes = plt.subplots(3, 3, figsize=(7,5))
    # axes = axes.flatten()

    # cmaps = ["Greys", "Blues", "Oranges", "Reds", "Purples", "Greens", "Dark2", "GnBu", "Pastel1"]
    # for i, (label, cmap) in enumerate(zip(labels, cmaps)):
    #     print(f"{i},{label},{cmap}")
    #     df_emb_sub = df_emb.query(f"label == {i}")
    #     axes[i].hexbin(df_emb_sub["X"], df_emb_sub["Y"], cmap=cmap, gridsize=20, linewidths=(0,))
    #     axes[i].set_title(label)
    #     axes[i].set_xticks([])
    #     axes[i].set_yticks([])

    # plt.tight_layout()
    # plt.show()


    ## If you want to train a dummy classifier to compare against
    # print("Training dummy classifier...")
    # lr_clf = DummyClassifier(strategy="most_frequent")
    # lr_clf.fit(X_train, y_train)
    
    # print("Testing dummy classifier")
    # score = lr_clf.score(X_test, y_test)
    # print(f"Got dummy score: {score}")
    
    ## Train logistic regression
    print("Training classifier...")
    lr_clf = LogisticRegression(max_iter=3000)
    lr_clf.fit(X_train, y_train)
    
    print("Saving classifier...")
    filename = 'lr_model.dat'
    pickle.dump(lr_clf, open(filename, 'wb'))
    
    print("Saving labels")
    with open('labels.txt', 'w') as f:
        for label in labels:
            f.write(label)
            f.write('\n')
    
    print("Testing classifier")
    score = lr_clf.score(X_test, y_test)
    print(f"Got score: {score}")
    
    y_preds = lr_clf.predict(X_test)
    plot_confusion_matrix(y_preds, y_test, labels)
    
    # TODO: Save LR model out and write predict script that can take input text and make predictions
    # TODO: Write script to spit out all model failures to another csv we can index into Oxen