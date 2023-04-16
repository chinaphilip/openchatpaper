from base_class import SimilarityAlg

from torch.nn import CosineSimilarity

class CosineSimilarity(SimilarityAlg):
    def __init__(self) -> None:
        pass

    @staticmethod
    def __call__(query_embedding, embeddings) -> None:
        return CosineSimilarity(query_embedding, embeddings)

