from retrieval_postgres.base_retriever import BasePostgresRetriever

class AutoMergePostgresRetriever(BasePostgresRetriever):
    def __init__(self, config):
        super().__init__(config, "auto_merge_postgres")
    def retrieve(self, query: str, top_k: int = 5):
        # TODO: implement auto-merge logic
        print("[AutoMergePostgresRetriever] (stub) returning empty result.")
        return [] 