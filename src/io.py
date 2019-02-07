import os
import nltk
from model.document import Document


def read_articles(path):
    articles = []
    cnt = 0

    for filename in os.listdir(path):
        with open(path + '/' + filename) as file:
            doc = Document(cnt, file.read())
            # doc.words = nltk.word_tokenize(doc.raw_text)
            # doc.sentences = nltk.sent_tokenize(doc.raw_text)
            cnt = cnt + 1
            articles.append(doc)

    return articles
