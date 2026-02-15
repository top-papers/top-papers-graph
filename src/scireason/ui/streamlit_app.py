from __future__ import annotations

import streamlit as st

from ..graph.graphrag_query import retrieve_context
from ..agents.debate_graph import run_debate

st.set_page_config(page_title="top-papers-graph", layout="wide")

st.title("top-papers-graph — Scientific Debate + GraphRAG (MVP)")

with st.sidebar:
    st.header("Настройки")
    collection = st.text_input("Qdrant collection", value="demo")
    domain = st.text_input("Домен/область", value="Science")
    k = st.slider("Сколько чанков доставать", 3, 20, 8)
    max_rounds = st.slider("Раундов дебатов", 1, 5, 3)

query = st.text_area("Запрос/задача", value="Найди противоречия в литературе про ... и предложи гипотезу")
run = st.button("Запустить")

col1, col2 = st.columns([1, 1])

if run:
    with st.spinner("Достаю контекст из Qdrant..."):
        ctx = retrieve_context(collection=collection, query=query, limit=int(k))
    context_text = "\n\n".join([f"[{c['payload'].get('paper_id')}] {c['payload'].get('text')}" for c in ctx])

    with col1:
        st.subheader("Контекст (выборка)")
        st.write(context_text)

    with st.spinner("Запускаю дебаты..."):
        result = run_debate(domain=domain, context=context_text, max_rounds=int(max_rounds))

    with col2:
        st.subheader("Гипотеза (после дебатов)")
        st.json(result.model_dump(mode="json"))

st.caption("MVP. Для продакшена используйте langgraph_debate.py и добавьте строгие схемы/валидаторы.")
