import streamlit as st
from typing import Dict, Any, Optional
import time
from datetime import datetime

from llama_index.core.schema import QueryBundle

from .cli_handlers import iLandRetrievalCLI

# Custom CSS for WhatsApp-like styling
WHATSAPP_CSS = """
<style>
    .stApp {background-color: #e5ddd5;}
    .user-message {
        background-color: #dcf8c6;
        padding: 8px 12px;
        border-radius: 7.5px;
        margin: 5px 0;
        max-width: 70%;
        float: right;
        clear: both;
        box-shadow: 0 1px 0.5px rgba(0,0,0,0.13);
    }
    .assistant-message {
        background-color: #ffffff;
        padding: 8px 12px;
        border-radius: 7.5px;
        margin: 5px 0;
        max-width: 70%;
        float: left;
        clear: both;
        box-shadow: 0 1px 0.5px rgba(0,0,0,0.13);
    }
    .message-time {
        font-size: 11px;
        color: #667781;
        margin-top: 4px;
    }
    .search-result {
        background-color: #f0f2f5;
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
        border-left: 3px solid #25d366;
    }
    .header {
        background-color: #075e54;
        color: white;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
</style>
"""

def initialize_session_state():
    if "cli" not in st.session_state:
        st.session_state.cli = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "is_typing" not in st.session_state:
        st.session_state.is_typing = False

def initialize_cli() -> Optional[iLandRetrievalCLI]:
    if st.session_state.cli is None:
        st.session_state.cli = iLandRetrievalCLI()
    cli = st.session_state.cli
    if not st.session_state.initialized:
        with st.spinner("\U0001F504 Initializing iLand system..."):
            try:
                if not cli.load_embeddings("latest"):
                    st.error("\u274C Failed to load embeddings")
                    return None
                if not cli.create_router("llm"):
                    st.error("\u274C Failed to create router")
                    return None
                st.session_state.initialized = True
                st.success("\u2705 System ready!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"\u274C Initialization error: {e}")
                return None
    return cli

def format_message_time(ts: Optional[datetime] = None) -> str:
    if ts is None:
        ts = datetime.now()
    return ts.strftime("%I:%M %p")

def display_search_results(results: list):
    if not results:
        return
    with st.container():
        st.markdown("**\U0001F4CD Found locations:**")
        for i, result in enumerate(results[:3]):
            col1, col2 = st.columns([3,1])
            with col1:
                st.markdown(f"**{i+1}.** {result.get('text','')[:100]}...")
            with col2:
                st.markdown(f"Score: {result.get('score',0):.2f}")
        if len(results) > 3:
            with st.expander(f"View all {len(results)} results"):
                for i, result in enumerate(results[3:],4):
                    st.markdown(
                        f"""<div class='search-result'><b>{i}. Score: {result.get('score',0):.3f}</b><br>{result.get('full_text','')}<br><small>Index: {result.get('index','N/A')} | Strategy: {result.get('strategy','N/A')}</small></div>""",
                        unsafe_allow_html=True,
                    )

def run_query_with_cli(cli: iLandRetrievalCLI, query: str, top_k: int) -> Dict[str, Any]:
    try:
        results = cli.query(query, top_k=top_k)
        response_text = None
        if cli.response_synthesizer:
            query_bundle = QueryBundle(query_str=query)
            nodes = cli.router._retrieve(query_bundle)
            response_text = cli.response_synthesizer.synthesize(query, nodes).response
        return {"results": results, "response": response_text, "success": True}
    except Exception as e:
        return {"results": [], "response": f"Sorry, I encountered an error: {e}", "success": False}

def main():
    st.set_page_config(page_title="iLand Chat", page_icon="\U0001F30F", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(WHATSAPP_CSS, unsafe_allow_html=True)
    initialize_session_state()
    st.markdown(
        """<div class='header'><h2>\U0001F30F iLand Land Deed Assistant</h2><small>Ask me about land deeds and property information</small></div>""",
        unsafe_allow_html=True,
    )
    cli = initialize_cli()
    if not cli:
        st.stop()
    with st.sidebar:
        st.header("\u2699\ufe0f Settings")
        top_k = st.slider("Number of results", 1, 10, 5)
        show_details = st.checkbox("Show technical details", value=False)
        if st.button("\U0001F5D1\ufe0f Clear chat"):
            st.session_state.messages = []
            st.rerun()
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(
                    f"<div class='user-message'>{message['content']}<div class='message-time'>{message.get('time','')}</div></div><div style='clear: both;'></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='assistant-message'>{message['content']}<div class='message-time'>{message.get('time','')}</div></div><div style='clear: both;'></div>",
                    unsafe_allow_html=True,
                )
                if show_details and message.get("results"):
                    display_search_results(message["results"])
    if st.session_state.is_typing:
        st.markdown("<div class='assistant-message'><i>typing...</i></div>", unsafe_allow_html=True)
    query = st.chat_input("Type a message...", key="chat_input")
    if query:
        current_time = format_message_time()
        st.session_state.messages.append({"role": "user", "content": query, "time": current_time})
        st.session_state.is_typing = True
        st.rerun()
    if st.session_state.is_typing and st.session_state.messages:
        last_message = st.session_state.messages[-1]
        if last_message["role"] == "user":
            output = run_query_with_cli(cli, last_message["content"], top_k)
            response_content = output.get("response") or f"I found {len(output['results'])} relevant land deed records for your query."
            st.session_state.messages.append({"role": "assistant", "content": response_content, "time": format_message_time(), "results": output["results"]})
            st.session_state.is_typing = False
            st.rerun()

if __name__ == "__main__":
    main()
