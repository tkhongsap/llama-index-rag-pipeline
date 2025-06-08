from typing import Dict, Any, List

import streamlit as st
from llama_index.core.schema import QueryBundle

from .cli_handlers import iLandRetrievalCLI


def initialize_cli() -> iLandRetrievalCLI:
    """Initialize the CLI and router if not already done."""
    if "cli" not in st.session_state:
        st.session_state.cli = iLandRetrievalCLI()
        st.session_state.initialized = False

    cli = st.session_state.cli

    if not st.session_state.initialized:
        with st.spinner("Loading embeddings and creating router..."):
            if not cli.load_embeddings("latest"):
                st.error("Failed to load embeddings")
                return cli
            if not cli.create_router("llm"):
                st.error("Failed to create router")
                return cli
        st.success("Router initialized")
        st.session_state.initialized = True
    return cli


def run_query(cli: iLandRetrievalCLI, query: str, top_k: int) -> Dict[str, Any]:
    """Run a query and return results and response."""
    query_bundle = QueryBundle(query_str=query)
    nodes = cli.router._retrieve(query_bundle)
    results = cli.operations._format_query_results(nodes, top_k)

    response_text = None
    if cli.response_synthesizer:
        try:
            response = cli.response_synthesizer.synthesize(query, nodes)
            response_text = response.response
        except Exception as e:
            response_text = f"Response generation failed: {e}"

    return {"results": results, "response": response_text}


st.set_page_config(page_title="iLand Retrieval", page_icon="🌏")

st.title("🌏 iLand Retrieval System")
cli = initialize_cli()

if "messages" not in st.session_state:
    st.session_state.messages = []  # chat history

# sidebar controls
top_k = st.sidebar.slider("Top K Results", 1, 10, 5)

# display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("results"):
            for res in msg["results"]:
                with st.expander(f"[{res['rank']}] Score: {res['score']:.3f}"):
                    st.write(res["full_text"])
                    st.write(
                        f"Index: {res['index']}\nStrategy: {res['strategy']}\n"
                        f"Index confidence: {res['index_confidence']:.2f} | "
                        f"Strategy confidence: {res['strategy_confidence']:.2f}"
                    )

# chat input
query = st.chat_input("Ask a land deed question")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    with st.spinner("Searching..."):
        output = run_query(cli, query, int(top_k))

    answer = output.get("response") or ""
    with st.chat_message("assistant"):
        st.markdown(answer or "No response")
        if output.get("results"):
            for res in output["results"]:
                with st.expander(f"[{res['rank']}] Score: {res['score']:.3f}"):
                    st.write(res["full_text"])
                    st.write(
                        f"Index: {res['index']}\nStrategy: {res['strategy']}\n"
                        f"Index confidence: {res['index_confidence']:.2f} | "
                        f"Strategy confidence: {res['strategy_confidence']:.2f}"
                    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "results": output.get("results"),
        }
    )

