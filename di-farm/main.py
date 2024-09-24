import json
import typing
import pandas as pd
import nltk
import torch
import random
from torch import nn
from sklearn import model_selection, preprocessing
from intent import IntentModel
from tokenizer import Tokenizer
from dataset import IntentDataset
from model import RnnModelV1
from engine import Engine

def load_intent_file():
    with open ("intents.json","r") as f:
        intent_json_blocks = json.load(f)
    
    intent_data=[]
    for block in intent_json_blocks["intents"]:
        intent = IntentModel(
            intent=block["intent"],
            text=block["text"],
            responses=block["responses"],
        )
        intent_data.append(intent)
    return intent_data

def create_dataset(intent_data:typing.List[IntentModel]) -> pd.DataFrame:
    intents=[]
    tags=[]

    for intent in intent_data:
        wx = [sent for sent in intent.text]
        intents.extend(wx)
        for _ in range(len(wx)):
            tags.append(intent.intent)
        
    dfx = (
            pd.DataFrame({"intents":intents,"tags":tags})
            .sample(frac=1)
            .reset_index(drop=True)
        )
    return dfx

def perform_prediction(
    model: RnnModelV1,
    tokenizer: Tokenizer,
    sentence: str,
    lbl_encoder: preprocessing.LabelEncoder,
    intent_data: typing.List[IntentModel],
):
    tokenized_dict = tokenizer.tokenize(sentence)
    token_type_ids = torch.tensor(
        tokenized_dict["token_type_ids"], dtype=torch.long
    ).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(token_type_ids)
        output = torch.softmax(output, dim=1)
        probs = output.max(dim=1)[0]
        output = output.argmax(dim=1)
        intent = lbl_encoder.inverse_transform(output.cpu().numpy())[0]
        roboresponse = [
            random.choice(intt.responses)
            for intt in intent_data
            if intt.intent == intent
        ][0]
        print(
            "[BOT]:",
            {
                "sentence": sentence,
                "predicted_intent": intent,
                "probability": round(probs.item(), 3),
                "bot_response": roboresponse,
            },
        )


def run():
    intent_class = load_intent_file()
    dfx = create_dataset(intent_class)
    label_encoder = preprocessing.LabelEncoder()
    dfx.loc[:,"encoded_tags"] = label_encoder.fit_transform(dfx.tags.values)
    dfx["fold"] = -1

    skf = model_selection.StratifiedKFold(n_splits=4, shuffle=True , random_state=0)

    for fold, (train_indices, test_indices) in enumerate(
        skf.split(dfx.drop("encoded_tags",axis=1), dfx["encoded_tags"].values)
        ):
            dfx.loc[test_indices,"fold"] = fold
    train_dfx, test_dfx = dfx.query("fold!=0").reset_index(drop=True), dfx.query("fold==0").reset_index(drop=True)
    print (dict(train=train_dfx.shape, test=test_dfx.shape))
    tokenizer = Tokenizer(texts=dfx["intents"].values,
                         pad_length=18)
    # print(tokenizer.tokenize("What's your real name?"))
    train_dataset = IntentDataset(
        intents=train_dfx["intents"].values,
        tags=train_dfx["encoded_tags"].values,
        tokenizer=tokenizer
    )
    test_dataset = IntentDataset(
        intents=test_dfx["intents"].values,
        tags=test_dfx["encoded_tags"].values,
        tokenizer=tokenizer
    )
    #  hyper parameters
    N_EMBEDDING_DIM = 512
    PADDING_IDX=123
    N_HIDDEN_LAYER=3
    N_HIDDEN_LAYER_NEURON = 512
    EPOCHS = 10
    BATCH_SIZE = 8
    LR = 3e-4

    model: nn.Module = RnnModelV1(
        n_embeddings=len(tokenizer.vocab_set) + 1,
        n_embedding_dim = N_EMBEDDING_DIM,
        padding_idx = PADDING_IDX,
        n_hidden_layer = N_HIDDEN_LAYER,
        n_hidden_layer_neurons = N_HIDDEN_LAYER_NEURON,
        n_classes=dfx["encoded_tags"].nunique(),
    )
    optimizer = torch.optim.Adam(model.parameters(),
                             lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        Engine.train(
            model=model,
            dataset=train_dataset,
            loss_fn=loss_fn,
            optimizer=optimizer,
            train_batch_size=BATCH_SIZE,
            epoch=epoch
        )
        Engine.test(
            model=model,
            dataset=train_dataset,
            loss_fn=loss_fn,
            train_batch_size=BATCH_SIZE,
            epoch=epoch
        )

    inp = input("Enter your sentence: ")
    while inp != "Q":
        perform_prediction(
            model=model,
            tokenizer=tokenizer,
            sentence=inp,
            lbl_encoder=label_encoder,
            intent_data=intent_class,
        )
        inp = input("[YOU]: ")





if __name__ == "__main__":
    # warnings.fi
    run()
