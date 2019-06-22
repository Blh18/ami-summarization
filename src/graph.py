import networkx as net
import nltk
import math
import nltk.stem.snowball as snowball
import nltk.classify.textcat as tc
from model.sentence import Sentence

stops = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
          "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
          "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs",
          "themselves", "what", "which", "who", "whom", "this", "that", "these", "those",
          "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
          "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or",
          "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
          "against", "between", "into", "through", "during", "before", "after", "above",
          "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
          "again", "further", "then", "once", "here", "there", "when", "where", "why", "how",
          "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no",
          "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can",
          "will", "just", "don", "should", "now"]

stemmer = snowball.EnglishStemmer()

text_classifier = tc.TextCat()

# create graph where nodes are sentences and edges are present if sentences are similar
def create_graph(sentences, words):
    graph = net.Graph()
    for s in sentences:
        graph.add_node(s.raw_text)
    graph = add_edges(graph, sentences, words)
    return graph

# create an edge in case the similarity of two sentences is above certain threshold
def add_edges(graph, sentences, words):
    for s1 in sentences:
        for s2 in sentences:
            if s1 == s2:
                continue
            similarity = count_cosine_similarity(s1, s2, len(sentences), words)
            if similarity > 0.7:
                graph.add_edge(s1.raw_text, s2.raw_text)
    return graph

# compute cosine similarity of two sentences using precomputed tfidf
def count_cosine_similarity(s1, s2, n, document):
    nominator = 0
    intersect = set(s1.words)
    intersect = intersect.intersection(s2.words)
    for w in intersect:
        tfs1 = s1.words[w]
        tfs2 = s2.words[w]
        idf = math.log10(n / document[w])
        nominator = nominator + (tfs1 * tfs2 * math.pow(idf, 2))
    denominator = math.sqrt(s1.tfidf) * math.sqrt(s2.tfidf)
    return nominator / denominator

# split text into sentences
# process each sentences to create bag of words in a sentence
def process_document(text):
    raw_sentences = nltk.sent_tokenize(text)
    sentences = []
    document = dict()

    for s in raw_sentences:
        words = nltk.word_tokenize(text_classifier.remove_punctuation(s))
        unique_words = set(words)
        sentence_dictionary = dict()

        increase_dict_value(unique_words, document)
        increase_dict_value(words, sentence_dictionary)

        sentences.append(Sentence(s, sentence_dictionary))

    return document, sentences

# inserts a word into a dictionary in case it is not a stop word
# stemmer is used to store just the root of word
def increase_dict_value(words, dictionary):
    for w in words:
        if w in stops:
            continue
        w = stemmer.stem(w)
        if w in dictionary:
            dictionary[w] = dictionary[w] + 1
        else:
            dictionary[w] = 1

# compute tfidf of all sentences in a document
def count_tfidf(sentences, document):
    for s in sentences:
        tf_idf_sum = 0
        for w, count in s.words.items():
            idf = math.log10(len(sentences) / document[w])
            tf = count
            tf_idf = tf * idf
            tf_idf_sum += math.pow(tf_idf, 2)
        s.tfidf = math.sqrt(tf_idf_sum)
    return 0
