import json
import numpy as np
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class UserKnowledgeBase:
    def __init__(self, user_id):
        self.user_id = user_id
        self.knowledge = {}
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self._ensure_fitted()

    def _ensure_fitted(self):
        if not self.knowledge:
            self.tfidf_vectorizer.fit(["dummy content for initialization"])
            self.tfidf_matrix = self.tfidf_vectorizer.transform(["dummy content for initialization"])

    def add_knowledge(self, topic: str, content: str):
        self.knowledge[topic] = content
        self._update_tfidf()

    def _update_tfidf(self):
        documents = list(self.knowledge.values())
        if documents:
            try:
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
            except ValueError:
                print("Warning: TfidfVectorizer fit failed. Using simple term frequency instead.")
                self.tfidf_matrix = np.array([[doc.count(word) for word in set(" ".join(documents).split())] for doc in documents])
        else:
            self._ensure_fitted()

    def get_relevant_knowledge(self, query: str, top_n: int = 3) -> List[str]:
        if not self.knowledge:
            return []

        try:
            query_vec = self.tfidf_vectorizer.transform([query])
        except ValueError:
            print("Warning: TfidfVectorizer transform failed. Using simple term frequency instead.")
            query_vec = np.array([[query.count(word) for word in set(" ".join(self.knowledge.values()).split())]])

        if isinstance(self.tfidf_matrix, np.ndarray):
            cosine_similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        else:
            cosine_similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        if len(self.knowledge) < top_n:
            top_n = len(self.knowledge)

        related_docs_indices = cosine_similarities.argsort()[:-top_n-1:-1]
        return [list(self.knowledge.values())[i] for i in related_docs_indices]

    def save(self):
        with open(f"{self.user_id}_knowledge.json", "w") as f:
            json.dump(self.knowledge, f)

    def load(self) -> bool:
        try:
            with open(f"{self.user_id}_knowledge.json", "r") as f:
                self.knowledge = json.load(f)
            self._update_tfidf()
            return True
        except FileNotFoundError:
            return False
