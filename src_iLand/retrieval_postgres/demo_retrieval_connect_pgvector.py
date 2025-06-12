from dotenv import dotenv_values
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.prompts import PromptTemplate, PromptType
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.storage.chat_store.postgres import PostgresChatStore
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.retrievers import QueryFusionRetriever
from colorama import Fore, Style, init

# Initialize colorama
init()

config = dotenv_values(".env")


def main():
    print(Fore.YELLOW + "Loading...\n\n" + Style.RESET_ALL)

    # set up
    base_url = config["ollama_api"]
    # remove trailing slash from
    if base_url.endswith("/"):
        base_url = base_url[:-1]
    ollama_embedding = OllamaEmbedding(
        model_name="bge-m3",
        base_url=base_url,
    )
    llm = Ollama(
        model="gemma3:27b",
        temperature=0.2,
        base_url=base_url,
        request_timeout=600,
        stream=True,
    )
    llm.complete("Hi")
    sub_query_llm = Ollama(
        model="gemma3:latest",
        temperature=0.2,
        base_url=base_url,
        request_timeout=600,
    )
    sub_query_llm.complete("Hi")

    db_name = "vector_db"
    index_table_name = "llamaindex_kbank_faqs"

    # configure vector store
    vector_store = PGVectorStore.from_params(
        database=db_name,
        host=config["pg_vector_host"],
        port=config["pg_vector_port"],
        user=config["pg_vector_user"],
        password=config["pg_vector_pass"],
        table_name=index_table_name,
        embed_dim=1024,
    )
    index = VectorStoreIndex.from_vector_store(
        vector_store, embed_model=ollama_embedding
    )

    # configure chat store
    chat_store = PostgresChatStore.from_params(
        host=config["pg_vector_host"],
        port=config["pg_vector_port"],
        user=config["pg_vector_user"],
        password=config["pg_vector_pass"],
        database=db_name,
        table_name="chat_store",
    )

    # custom prompt template
    template = """Use the following pieces of context to answer the question at the end.
    ONLY USE THE CONTEXT PROVIDED. DO NOT ADD ANY ADDITIONAL INFORMATION.
    Answer the question as if you were a human expert in the field.

    {context_str}
    """
    prompt_template = PromptTemplate(template)

    QUERY_GEN_PROMPT = (
        "You are a helpful assistant that generates multiple search queries based on a single input query. "
        "Try to break down the input query into multiple sub-questions to help answer the input query. "
        "Generate {num_queries} search queries, one on each line, "
        "related to the following input query:\n"
        "Query: {query}\n"
        "Queries:\n"
    )

    # configure retriever
    retriever = QueryFusionRetriever(
        [index.as_retriever()],
        similarity_top_k=5,
        num_queries=4,  # set this to 1 to disable query generation
        use_async=True,
        llm=sub_query_llm,
        query_gen_prompt=QUERY_GEN_PROMPT,  # we could override the query generation prompt here
    )

    # configure chat memory
    chat_store_key = input("Enter user id: ")
    chat_memory = ChatMemoryBuffer.from_defaults(
        token_limit=3000,
        chat_store=chat_store,
        chat_store_key=chat_store_key,
    )

    # assemble chat engine
    chat_engine = ContextChatEngine(
        retriever=retriever,
        llm=llm,
        memory=chat_memory,
        prefix_messages=[],
        context_template=prompt_template,
        node_postprocessors=[
            MetadataReplacementPostProcessor(target_metadata_key="window")
        ],
    )

    while True:
        query = input("\nEnter query (s for stop, c for clear): ")
        if query in ["s", "stop"]:
            break
        if query in ["c", "clear"]:
            chat_memory.reset()
            print(Fore.YELLOW + "Memory cleared.\n" + Style.RESET_ALL)
            continue
        print(Fore.YELLOW + f"question: {query}\n" + Style.RESET_ALL)

        # Create a custom response callback that prints in yellow
        class YellowResponseCallback:
            def __call__(self, token):
                print(Fore.YELLOW + token + Style.RESET_ALL, end="", flush=True)

        response = chat_engine.stream_chat(query)
        # Use our custom callback instead of the default print_response_stream
        for token in response.response_gen:
            print(Fore.YELLOW + token + Style.RESET_ALL, end="", flush=True)

        print(Fore.YELLOW + "\n\n" + Style.RESET_ALL)


if __name__ == "__main__":
    main()
