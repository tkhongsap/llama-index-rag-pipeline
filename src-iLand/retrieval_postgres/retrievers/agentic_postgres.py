from retrieval_postgres.base_retriever import BasePostgresRetriever

class AgenticPostgresRetriever(BasePostgresRetriever):
    def __init__(self, config):
        super().__init__(config, "agentic_postgres")
    def retrieve(self, query: str, top_k: int = 5):
        # TODO: implement agentic logic
        print("[AgenticPostgresRetriever] (stub) returning empty result.")
        return [] 