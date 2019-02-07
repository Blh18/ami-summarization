class Sentence:
    def __init__(self, text, words):
        self.raw_text = text
        self.words = words
        self.tfidf = None
