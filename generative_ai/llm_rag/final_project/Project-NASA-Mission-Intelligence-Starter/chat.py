#!/usr/bin/env python3
"""Interactive NASA Mission Intelligence RAG application."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import streamlit as st

import llm_client
import rag_client
import ragas_evaluator


st.set_page_config(page_title="NASA Mission Intelligence", page_icon="🚀", layout="wide")


@st.cache_data(ttl=10)
def discover_chroma_backends() -> Dict[str, Dict[str, Any]]:
    return rag_client.discover_chroma_backends()


@st.cache_resource
def initialize_rag_system(
    chroma_dir: str, collection_name: str, openai_key: str
):
    return rag_client.initialize_rag_system(
        chroma_dir, collection_name, openai_api_key=openai_key
    )


def display_evaluation_metrics(scores: Dict[str, Any]) -> None:
    st.sidebar.subheader("Response quality")
    if "error" in scores:
        st.sidebar.warning(str(scores["error"]))
        return
    for metric_name, score in scores.items():
        if isinstance(score, (int, float)):
            st.sidebar.metric(metric_name.replace("_", " ").title(), f"{score:.3f}")
            st.sidebar.progress(max(0.0, min(1.0, float(score))))


def _render_sources(message: Dict[str, Any]) -> None:
    sources = message.get("sources") or []
    if not sources:
        return
    with st.expander(f"Retrieved evidence ({len(sources)} excerpts)"):
        for index, source in enumerate(sources, 1):
            metadata = source.get("metadata", {})
            distance = source.get("distance")
            label = (
                f"Source {index}: {metadata.get('source', 'unknown')} · "
                f"{str(metadata.get('mission', 'unknown')).replace('_', ' ').title()}"
            )
            if isinstance(distance, (int, float)):
                label += f" · distance {distance:.4f}"
            st.markdown(f"**{label}**")
            st.caption(source.get("document", ""))


def main() -> None:
    st.title("🚀 NASA Mission Intelligence")
    st.caption(
        "Source-grounded answers from Apollo 11, Apollo 13, and Challenger archive documents"
    )
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_evaluation", None)

    with st.sidebar:
        st.header("Configuration")
        available_backends = discover_chroma_backends()
        usable_backends = {key: value for key, value in available_backends.items() if not value.get("error")}
        if not usable_backends:
            st.error("No indexed ChromaDB collection was found.")
            st.code(
                "python embedding_pipeline.py --data-path ./data_text "
                "--chroma-dir ./chroma_db_openai"
            )
            st.stop()

        selected_key = st.selectbox(
            "Document collection",
            list(usable_backends),
            format_func=lambda key: usable_backends[key]["display_name"],
        )
        backend = usable_backends[selected_key]
        openai_key = st.text_input(
            "OpenAI API key", type="password", value=os.getenv("OPENAI_API_KEY", "")
        )
        model = st.selectbox("Answer model", ["gpt-4.1-mini", "gpt-4.1"])
        mission_label = st.selectbox(
            "Mission scope", ["All missions", "Apollo 11", "Apollo 13", "Challenger"]
        )
        mission_filter = {
            "All missions": None,
            "Apollo 11": "apollo_11",
            "Apollo 13": "apollo_13",
            "Challenger": "challenger",
        }[mission_label]
        n_docs = st.slider("Excerpts to retrieve", 1, 10, 4)
        enable_evaluation = st.checkbox(
            "Run RAGAS evaluation", value=False, help="Uses additional OpenAI calls."
        )
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_evaluation = None
            st.rerun()
        if st.session_state.last_evaluation and enable_evaluation:
            display_evaluation_metrics(st.session_state.last_evaluation)

    if not openai_key:
        st.info("Enter an OpenAI API key in the sidebar to query the archive.")
        st.stop()
    os.environ["OPENAI_API_KEY"] = openai_key
    os.environ["CHROMA_OPENAI_API_KEY"] = openai_key

    collection, success, error = initialize_rag_system(
        backend["directory"], backend["collection_name"], openai_key
    )
    if not success:
        st.error(f"Could not open the selected collection: {error}")
        st.stop()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                _render_sources(message)

    prompt = st.chat_input("Ask a question about a NASA mission…")
    if not prompt:
        return
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving NASA evidence and composing an answer…"):
                result = rag_client.retrieve_documents(
                    collection, prompt, n_results=n_docs, mission_filter=mission_filter
                )
                documents = (result or {}).get("documents", [[]])[0]
                metadatas = (result or {}).get("metadatas", [[]])[0]
                distances = (result or {}).get("distances", [[]])[0]
                context = rag_client.format_context(documents, metadatas, distances)
                history = [
                    {"role": item["role"], "content": item["content"]}
                    for item in st.session_state.messages[:-1]
                ]
                response = llm_client.generate_response(
                    openai_key, prompt, context, history, model=model
                )
            st.markdown(response)
            sources = [
                {"document": doc, "metadata": meta, "distance": distance}
                for doc, meta, distance in zip(documents, metadatas, distances)
            ]
            assistant_message = {"role": "assistant", "content": response, "sources": sources}
            _render_sources(assistant_message)
            st.session_state.messages.append(assistant_message)

            if enable_evaluation:
                with st.spinner("Calculating RAGAS metrics…"):
                    st.session_state.last_evaluation = ragas_evaluator.evaluate_response_quality(
                        prompt, response, list(documents), openai_api_key=openai_key
                    )
                display_evaluation_metrics(st.session_state.last_evaluation)
        except Exception as exc:
            error_message = f"Unable to answer this question: {exc}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})


if __name__ == "__main__":
    main()
