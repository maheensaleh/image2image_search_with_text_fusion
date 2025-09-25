import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

def compute_similarity(query_embedding, embeddings, similarity_metric='cosine', top_k=None):

    if similarity_metric == 'cosine':
        similarity_scores = cosine_similarity([query_embedding], embeddings)
    elif similarity_metric == 'euclidean':
        similarity_scores = -euclidean_distances([query_embedding], embeddings)
    else:
        raise ValueError("Invalid similarity metric. Choose 'cosine' or 'euclidean'.")

    similarity_scores = similarity_scores.flatten()  # Flatten the similarity scores

    # Get the indices of the top K similar embeddings
    if top_k is None:
      top_indices = np.argsort(similarity_scores)[::-1]
    else: 
      top_indices = np.argsort(similarity_scores)[::-1][:top_k]

    # Create a list of tuples containing the top similar embeddings and their similarity scores
    top_similar_embeddings = [(embeddings[idx], similarity_scores[idx], idx) for idx in top_indices]

    return top_similar_embeddings


if __name__ == "__main__":  

    # Example query embedding
    query_embedding = np.array([0.8, 0.6, 0.2])

    # Example list of embeddings
    embeddings = np.array([
        [0.8, 0.6, 0.2],
        [0.7, 0.3, 0.5],
        [0.4, 0.1, 0.9],
        [0.2, 0.9, 0.7]
    ])

    # Compute and print the top similar embeddings
    top_k_similar = compute_similarity(query_embedding, embeddings, similarity_metric='cosine', top_k=2)
    for embedding, similarity,idx in top_k_similar:
        print("--------{}---------".format(idx))
        print(f"Similarity score: {similarity}")
        print(f"Similar embedding: {embedding}")
