
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

def predict(raw_args):
    global device, tokenizer

    parser = argparse.ArgumentParser(
        description="Command line tool to train a text classification model off of an input dataset"
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="The input model file",
    )
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        required=True,
        help="The input file of data",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        required=True,
        help="The output file of the predictions",
    )
    args = parser.parse_args(raw_args)
    
    # dataset = load_dataset("csv", data_files=args.input_file, sep='\t', names=["text", "label"])
    
    df = pd.read_csv(args.input_file, on_bad_lines='warn', delimiter='\t')    
    dataset = DatasetDict()
 
    split_on = int(df.shape[0] * 0.8)
    dataset['test'] = Dataset.from_pandas(df[split_on:])
    
    # Convert strings to labels
    dataset = dataset.class_encode_column("label")

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
    
    X_test = np.array(dataset_hidden['test']['hidden_state'])
    y_test = np.array(dataset_hidden['test']['label'])

    print(f"X_test.shape: {X_test.shape}")
    labels = dataset["test"].features["label"].names
    print(labels)

    ## Load logistic regression
    print("Loading classifier...")
    lr_clf = pickle.load(open(args.model, 'rb'))
    
    print("Testing classifier")
    score = lr_clf.score(X_test, y_test)
    print(f"Got score: {score}")
    
    y_preds = lr_clf.predict_proba(X_test)
    with open(args.output_file, 'w') as f:
        f.write(f"index\tcorrect_label\tcorrect_label_idx\tpred_label\tpred_label_idx\tis_correct\tprobability\ttext\n")
        for i, preds in enumerate(y_preds):
            print(preds)
            pred_idx = np.argmax(preds)
            pred_proba = preds[pred_idx]
            
            example = dataset['test'][i]
            correct = example['label']
            line = (f"{i}\t{labels[correct]}\t{correct}\t{labels[pred_idx]}\t{pred_idx}\t{correct==pred_idx}\t{pred_proba}\t{example['text']}")
            f.write(line)
            f.write('\n')
