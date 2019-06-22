from src.io import read_articles
import matplotlib.pyplot as plt
import networkx as net

# NOTES:
# - many use cases for summarization - web, e-mails, documents, ...
# - summarization saves time and might help a user to make a decision faster
# - Multi document summarization = compute relations in text, order the items (words, sentences), return a portion
# - graph based methods - nodes are words, sentences and edges represent ties between them
# 1st part - PROCESSING THE TEXT
# - split the text into sentences
# - process each sentence to get bag of words - stemming, stop words
# 2nd part - COMPUTE SIMILARITIES
# - pick a similarity measure - jaccard, subsequence, cosine similarity, substrings, ...
# 3rd part - SELECT SENTENCES FOR SUMMARY
# - different ways to select sentences for summaries
# - we used random walk - graph - nodes, edges and employ page rank
# - edges only if the sentences are above some similarity threshold
# - page rank: we have a probability matrix that we use for power iteration
# - page rank: teleportation so we don't get stuck
# - page rank: similarity matrix would not work, we just add edge if similarity is above threshold

# LIMITATIONS:
# - we do not take different elements into account (headings, side notes, ...)
# - sentences tend to cluster around sub-topics
# - use some cluster detection algorithm (k-means)
# - use the information about clusters to increase/decrease transition probabilities



# process all available articles
articles = read_articles("./articles")

# display graph of the sentences in an article
net.draw(articles[0].graph)
plt.show()

# run page rank on the graph
v = net.pagerank(articles[0].graph)

# sort sentences of an article by their page rank
s = sorted(v.items(), key=lambda p: p[1], reverse=True)

# print first two sentences of an article
print(s[0][0] + ' ' + s[1][0])
