import nltk
import typing
import string

class Tokenizer:
    def __init__(self,texts:typing.List[str],pad_length:int=20) -> None:
        self.___rawtexts = texts.copy()
        self.pad_length = pad_length
        self.__lemmer = nltk.stem.WordNetLemmatizer()
        self.texts = [" ".join(self.___tokenize(sent)) for sent in self.___rawtexts] # this will join the tokens of a sentence into one single string separated by space and then add all those sentences in a list
        self.vocab_set = self.__build_vocab(self.texts)
        self.word_to_idx = dict(
          (word,index)  for index, word in enumerate(self.vocab_set)
        )
        self.idx_to_word = dict(
            (index,word) for index, word in enumerate(self.vocab_set)
        )
    
    def ___tokenize(self,sentence:str):
        words = nltk.word_tokenize(sentence.lower().strip())[:self.pad_length]
        paper = ["[SEP]"]
        for word in words:
            if not word in set(string.punctuation) and not word in ["[UNK]","[PAD]"]:
                paper.append(self.__lemmer.lemmatize(word))
        paper.append("[SEP]")
        paper.extend(["[PAD]"] * (self.pad_length - len(paper))) # appends multiple pads at once
        return paper
    
    def __build_vocab(self, texts: typing.List[str]):
        vocab_set = set()

        for text in texts:
            for sent in nltk.sent_tokenize(text):
                for word in nltk.word_tokenize(sent):
                    vocab_set.add(word)

        vocab_set.add("[PAD]")
        vocab_set.add("[SEP]")
        vocab_set.add("[UNK]")
        vocab_set.add("[CLS]")
        
        return vocab_set
