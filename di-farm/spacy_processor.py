import spacy
import string

# Load spaCy's language model
nlp = spacy.load("en_core_web_md")

class SpacyProcessor:
    def __init__(self):
        self.nlp = nlp
    
    def process_text(self, text: str):
        # Create the Doc object once
        doc = self.nlp(text.lower().strip())
        return doc
    
    def tokenize(self, doc):
        # doc = nlp(sentence)
        return [token.text for token in doc]
    
    def lemmatize(self, word):
        word=word.lower()
        doc = nlp(word)
        return doc[0].lemma_

# # Example usage
# processor = TextProcessor()

# # Create the doc once
# doc = processor.process_text("Running is fun.")

# # Reuse the doc in different functions
# print("Tokens:", processor.tokenize(doc))
# print("Lemmas:", processor.lemmatize(doc))
