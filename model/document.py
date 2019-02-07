class Document:
    def __init__(self, doc_id, text):
        self.id = doc_id
        self.raw_text = text.replace('\n', ' ')
        self.score = 0
        self.page_rank = 1
        self.words = dict()
        self.sentences = dict()
        self.graph = None
