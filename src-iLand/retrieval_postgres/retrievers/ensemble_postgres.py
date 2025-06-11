from retrieval_postgres.base_retriever import BasePostgresRetriever

class EnsemblePostgresRetriever(BasePostgresRetriever):
    def __init__(self, config):
        super().__init__(config, "ensemble_postgres")
    def retrieve(self, query: str, top_k: int = 5):
        # TODO: implement ensemble logic
        print("[EnsemblePostgresRetriever] (stub) returning empty result.")
        return [] 