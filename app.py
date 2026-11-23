"""Interface Streamlit simples para o agente GeoValor (DR3 - TP3)."""

from __future__ import annotations

import os
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

from agente import criar_agente, perguntar
from ferramentas import ErroDados, carregar_terrenos

load_dotenv()
st.set_page_config(page_title="GeoValor | DR3 TP3", page_icon="🏡", layout="centered")
st.title("🏡 GeoValor — Consulta de Terrenos")
st.caption("TP3 | Qwen via Groq • LangChain • 2 ferramentas • ReAct • memória")
st.info("Base estática de anúncios de Joaçaba/SC. Preços de oferta, não valores de transações ou laudos.")

# Session State mantém a visualização; o InMemorySaver mantém o estado do agente.
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid4())
if "historico" not in st.session_state:
    st.session_state.historico = []


@st.cache_resource(show_spinner=False)
def iniciar_agente():
    """Evita reconstruir o agente e perder a memória a cada interação."""
    return criar_agente()


with st.sidebar:
    st.header("Dados e conversa")
    try:
        terrenos = carregar_terrenos()
        st.metric("Anúncios no CSV", len(terrenos))
        st.caption("Fontes: SOMAR, Pedro Fiório e Ivam Rover.")
        st.caption("Base fechada: não realiza scraping ou atualização automática.")
        with st.expander("Consultar a base original"):
            st.dataframe(terrenos, use_container_width=True, hide_index=True)
    except ErroDados as exc:
        st.error(str(exc))
    if st.button("Nova conversa", use_container_width=True):
        st.session_state.thread_id = str(uuid4())
        st.session_state.historico = []
        st.rerun()

st.markdown("**Experimente:** 'Quantos terrenos há?', 'Quais são do Jardim Cidade Alta?' ou 'Calcule a mediana do preço por m² dos terrenos encontrados'.")

for item in st.session_state.historico:
    with st.chat_message(item["papel"]):
        st.markdown(item["texto"])
        if item.get("trilha"):
            with st.expander("Ações e observações (ReAct)"):
                for passo in item["trilha"]:
                    st.json(passo)

pergunta = st.chat_input("Pergunte sobre os 26 anúncios...")
if pergunta:
    st.session_state.historico.append({"papel": "user", "texto": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)
    with st.chat_message("assistant"):
        trilha = []
        if not os.getenv("GROQ_API_KEY"):
            resposta = "Configure GROQ_API_KEY no arquivo .env. As instruções estão no README.md."
            st.warning(resposta)
        else:
            try:
                with st.spinner("Consultando os dados..."):
                    resposta, trilha = perguntar(iniciar_agente(), pergunta, st.session_state.thread_id)
                st.markdown(resposta)
                if trilha:
                    with st.expander("Ações e observações (ReAct)", expanded=True):
                        for passo in trilha:
                            st.json(passo)
            except Exception as exc:
                tipo = type(exc).__name__
                if tipo in {"AuthenticationError", "PermissionDeniedError"}:
                    resposta = "Chave Groq inválida ou sem permissão. Confira GROQ_API_KEY."
                elif tipo == "RateLimitError":
                    resposta = "Limite de requisições do Groq atingido. Tente novamente depois."
                elif tipo == "BadRequestError":
                    resposta = "A API recusou a solicitação. Confira GROQ_MODEL e o suporte a ferramentas."
                elif tipo in {"APITimeoutError", "APIConnectionError"}:
                    resposta = "Não foi possível comunicar com o GroqCloud. Verifique a conexão."
                elif isinstance(exc, (ErroDados, ValueError)):
                    resposta = f"Não foi possível consultar os dados: {exc}"
                else:
                    resposta = f"Falha ao executar o agente ({tipo}). Confira o terminal para detalhes."
                st.error(resposta)
        st.session_state.historico.append({"papel": "assistant", "texto": resposta, "trilha": trilha})
