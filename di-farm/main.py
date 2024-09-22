import json
import typing
import pandas as pd
import nltk
from sklearn import model_selection, preprocessing
from intent import IntentModel
from tokenizer import Tokenizer

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
    tokenize = Tokenizer(texts=dfx["intents"].values,
                         pad_length=18)
    tokenize






if __name__ == "__main__":
    # warnings.fi
    run()
