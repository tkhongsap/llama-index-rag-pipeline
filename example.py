import os

from dotenv import load_dotenv
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import Node
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.postgres import PGVectorStore


class Indexer:
    def __init__(
        self,
        embed_model: BaseEmbedding = None,
        vector_store: BasePydanticVectorStore = None,
        db_config: dict = None,
        base_url: str = None,
    ):
        """
        Initializes the Indexer class.

        Args:
            embed_model (BaseEmbedding, optional): The embedding model to use. Defaults to None.
            vector_store (BasePydanticVectorStore, optional): The vector store to use. Defaults to None.
            db_config (dict, optional): The database configuration. Defaults to None.
            base_url (str, optional): The base URL for the embedding model. Defaults to None.
        """
        self.embed_model = embed_model or self._get_default_embed_model(base_url)
        self.vector_store = vector_store or self.load_pg_vector_store(db_config)
        self.index = self.load_index_from_vector_store(
            self.vector_store, self.embed_model
        )

    def get_index(self):
        return self.index

    def load_pg_vector_store(self, db_config):
        """
        Loads a PGVectorStore instance using the provided database configuration.

        Args:
            db_config (dict): A dictionary containing the database configuration with the following keys:
                - host (str): The database host.
                - port (int): The database port.
                - username (str): The database username.
                - password (str): The database password.
                - database (str): The name of the database.
                - table_name (str): The name of the table to use.
                - embed_dim (int): The embedding dimension.

        Returns:
            PGVectorStore: An instance of PGVectorStore configured with the provided parameters.
        """
        vector_store = PGVectorStore.from_params(
            host=db_config["host"],
            port=db_config["port"],
            user=db_config["username"],
            password=db_config["password"],
            database=db_config["database"],
            table_name=db_config["table_name"],
            embed_dim=db_config["embed_dim"],
        )
        return vector_store

    def load_index_from_vector_store(self, vector_store, embed_model):
        """
        Loads an index from a given vector store using the specified embedding model.

        Args:
            vector_store: The vector store from which to load the index.
            embed_model: The embedding model to use for loading the index.

        Returns:
            The loaded index.
        """
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store, embed_model=embed_model
        )
        return self.index

    def add_nodes_to_index(self, nodes: list[Node], **kwargs):
        self.index.insert_nodes(nodes, **kwargs)

    def remove_nodes_from_index(self, node_ids, **kwargs):
        self.index.delete_nodes(node_ids, **kwargs)

    def retrive(self, query: str, top_k=5):
        return self.index.as_retriever(similarity_top_k=top_k).retrieve(query)

    def _get_default_embed_model(self, base_url=None):
        base_url = base_url or os.getenv("ollama_api", "http://localhost:11434")
        embed_model = OllamaEmbedding(model_name="bge-m3", base_url=base_url)
        return embed_model


if __name__ == "__main__":
    load_dotenv()

    config = {
        "database": os.getenv("pg_vector_db_name", "postgres"),
        "host": os.getenv("pg_vector_host", "localhost"),
        "port": os.getenv("pg_vector_port", 5432),
        "username": os.getenv("pg_vector_user", "postgres"),
        "password": os.getenv("pg_vector_pass", "password"),
        "table_name": "llamaindex_kbank_faqs",
        "embed_dim": 1024,
    }
    indexer = Indexer(db_config=config, base_url=os.getenv("ollama_api"))
    nodes = indexer.retrive("How to open a bank account?")
    for node in nodes:
        print("Node ID:", node.node_id)
        print("Node Text:", node.text[:100] + "...")
        print("=====================================")
