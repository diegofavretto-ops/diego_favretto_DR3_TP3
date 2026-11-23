"""Agente LangChain que escolhe duas ferramentas e mantém memória temporária."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver

from ferramentas import criar_ferramentas

load_dotenv()

INSTRUCOES = """Você é o GeoValor, um agente educacional de consulta de terrenos de Joaçaba/SC.
Os dados vêm EXCLUSIVAMENTE do CSV de 26 anúncios de SOMAR, Pedro Fiório e Ivam Rover.

Regras:
1. Para toda pergunta factual sobre os terrenos, chame buscar_terrenos. Não invente preços,
   áreas, bairros, quantidades, URLs ou anúncios não encontrados.
2. Para perguntas que exigem média, mediana, mínimo, máximo, comparação por bairro ou
   estatísticas, primeiro busque os terrenos e DEPOIS chame analisar_precos, enviando
   exatamente a lista de códigos encontrada na busca; nunca calcule sozinho.
3. Quando a pergunta se referir a 'deles', 'nesse bairro' etc., use a memória da conversa
   para preservar o bairro, a imobiliária e filtros anteriores. Na dúvida, pergunte.
4. Se a busca não encontrar resultados, informe isso. Não chame a análise com lista vazia.
5. Se o usuário pedir coleta ao vivo ou atualização, explique que a versão do TP3 usa
   um CSV fechado; não existe scraping nem atualização online neste projeto.
6. Informe a quantidade de registros utilizada, destaque a fonte e inclua URLs ao citar
   exemplos de anúncios. Não confunda preço total e preço por metro quadrado.
7. São valores de oferta, não preços efetivos de venda. Não emita laudo de avaliação.
8. Responda em português claro. A tela pode mostrar ações/observações das ferramentas;
   nunca invente chamadas realizadas.
"""


def criar_agente(modelo=None, memoria=None):
    """Monta o agente e permite injetar LLM de teste sem consumir a API."""
    if modelo is None:
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("Defina GROQ_API_KEY no arquivo .env antes de usar o GeoValor.")
        modelo = ChatGroq(
            model=os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
            temperature=0.2,
            max_retries=1,
        )
    return create_agent(
        model=modelo,
        tools=criar_ferramentas(),
        system_prompt=INSTRUCOES,
        checkpointer=memoria if memoria is not None else InMemorySaver(),
    )


def perguntar(agente, pergunta: str, thread_id: str) -> tuple[str, list[dict]]:
    """Retorna resposta e trilha de ações/observações da pergunta atual."""
    if not pergunta.strip():
        raise ValueError("Digite uma pergunta para o agente.")
    if not thread_id.strip():
        raise ValueError("Identificador da conversa ausente.")
    estado = agente.invoke(
        {"messages": [{"role": "user", "content": pergunta}]},
        config={"configurable": {"thread_id": thread_id}, "recursion_limit": 16},
    )
    mensagens = estado["messages"]
    # Retém apenas as ferramentas usadas após a última mensagem humana.
    inicio = max((i for i, m in enumerate(mensagens) if m.type == "human"), default=-1)
    trilha = []
    for mensagem in mensagens[inicio + 1:]:
        if mensagem.type == "ai":
            for chamada in getattr(mensagem, "tool_calls", []) or []:
                trilha.append({"etapa": "ação", "ferramenta": chamada["name"],
                              "parâmetros": chamada.get("args", {})})
        elif mensagem.type == "tool":
            trilha.append({"etapa": "observação", "ferramenta": mensagem.name,
                          "resultado": str(mensagem.content)[:1500]})
    if not mensagens or mensagens[-1].type != "ai":
        raise RuntimeError("O agente não produziu uma resposta final.")
    resposta = mensagens[-1].content
    if isinstance(resposta, list):
        resposta = "\n".join(parte.get("text", "") for parte in resposta if isinstance(parte, dict))
    return str(resposta), trilha
