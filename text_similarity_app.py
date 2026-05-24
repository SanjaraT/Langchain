from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from documents import documents

# loading embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

# documents --> embeddings
document_embeddings = embedding_model.embed_documents(documents)

# query 
query = "What model is useful for text understanding?"

# query --> embedding
query_embedding = embedding_model.embed_query(query)

# cosine similarity
similarities = cosine_similarity(
    [query_embedding],
    document_embeddings
)[0]

# matching document
similar_index = np.argmax(similarities)

print(query)
print(documents[similar_index])
print("\nSimilarity Score : ")
print(similarities[similar_index])


