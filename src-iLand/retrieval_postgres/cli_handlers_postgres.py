import sys
from retrieval_postgres.config import PostgresConfig
from retrieval_postgres.router import PostgresRouterRetriever

class PostgresRetrievalCLI:
    def __init__(self):
        self.config = PostgresConfig.from_env()
        self.router = PostgresRouterRetriever(self.config)

    def create_router(self):
        print(f"Router set to: PostgresRouterRetriever (auto strategy)")
        return True

    def query(self, query: str, top_k: int = 5):
        if not self.router:
            print("Router not initialized.")
            return
        print(f"\nQuery: {query}")
        try:
            results = self.router.retrieve(query, top_k=top_k)
            print(f"\nTop {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"\n--- Result {i} ---")
                print(f"Score: {getattr(result, 'score', 0):.4f}")
                print(f"Text: {getattr(result, 'text', '')[:200]}...")
                if hasattr(result, 'metadata') and result.metadata:
                    print(f"Metadata: {result.metadata}")
        except Exception as e:
            if "does not exist" in str(e) or "UndefinedTable" in str(e):
                print("❗ ตาราง embedding (เช่น data_iland_chunks) ยังไม่ถูกสร้างใน database นี้\nโปรดแจ้งทีม data/infra ให้สร้างตารางก่อนใช้งาน retrieval/query")
            else:
                print(f"Error: {e}")

    def test_strategies(self, queries, top_k=5):
        for query in queries:
            self.query(query, top_k=top_k)

    def interactive_mode(self, top_k=5):
        print("=== iLand PostgreSQL Interactive Retrieval CLI ===")
        print("Type 'exit' to quit.")
        while True:
            try:
                query = input("\nEnter your query: ").strip()
                if query.lower() == "exit":
                    break
                self.query(query, top_k=top_k)
            except KeyboardInterrupt:
                print("\nExiting.")
                break
            except Exception as e:
                print(f"Error: {e}") 