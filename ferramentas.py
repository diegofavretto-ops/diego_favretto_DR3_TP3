"""Consulta e análise de anúncios: duas ferramentas para o agente LangChain.

Este módulo mantém os cálculos separados da LLM. O decorador @tool é
aplicado em criar_ferramentas(), permitindo testar os dados sem chamar API.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

ARQUIVO_CSV = Path(__file__).resolve().parent / "dados" / "terrenos.csv"
COLUNAS = {"codigo", "bairro", "cidade", "area_m2", "preco_brl", "url", "fonte"}


class ErroDados(Exception):
    """Informa problemas de leitura ou validação da base de terrenos."""


def _normalizar(texto: str) -> str:
    """Permite comparar bairros ignorando acentos, caixa e espaços."""
    texto = unicodedata.normalize("NFKD", str(texto))
    return "".join(caractere for caractere in texto if not unicodedata.combining(caractere)).casefold().strip()


def carregar_terrenos(caminho: Path = ARQUIVO_CSV) -> list[dict]:
    """Lê o CSV e retorna registros válidos; rejeita linhas inconsistentes."""
    try:
        with Path(caminho).open("r", encoding="utf-8-sig", newline="") as arquivo:
            leitor = csv.DictReader(arquivo)
            if not leitor.fieldnames or not COLUNAS.issubset(leitor.fieldnames):
                raise ErroDados("O CSV não possui todas as colunas obrigatórias.")
            terrenos = []
            vistos = set()
            for linha_numero, linha in enumerate(leitor, start=2):
                try:
                    codigo = str(linha["codigo"] or "").strip()
                    bairro = str(linha["bairro"] or "").strip()
                    fonte = str(linha["fonte"] or "").strip()
                    cidade = str(linha["cidade"] or "").strip()
                    url = str(linha["url"] or "").strip()
                    area = float(linha["area_m2"])
                    preco = float(linha["preco_brl"])
                except (ValueError, TypeError, KeyError) as exc:
                    raise ErroDados(f"Valores inválidos na linha {linha_numero} do CSV.") from exc
                if not codigo or not bairro or not fonte or not cidade:
                    raise ErroDados(f"Identificação incompleta na linha {linha_numero}.")
                if not math.isfinite(area) or not math.isfinite(preco) or area <= 0 or preco <= 0:
                    raise ErroDados(f"Preço ou área inválidos na linha {linha_numero}.")
                if urlparse(url).scheme not in ("http", "https") or not urlparse(url).netloc:
                    raise ErroDados(f"URL inválida na linha {linha_numero}.")
                chave = (fonte.casefold(), codigo.casefold())
                if chave in vistos:
                    raise ErroDados(f"Referência duplicada no CSV: {fonte} / {codigo}.")
                vistos.add(chave)
                terrenos.append({
                    "codigo": codigo,
                    "bairro": bairro,
                    "cidade": cidade,
                    "area_m2": area,
                    "preco_brl": preco,
                    "preco_m2_brl": round(preco / area, 2),
                    "url": url,
                    "fonte": fonte,
                    "consultado_em": str(linha.get("consultado_em") or "").strip(),
                })
    except OSError as exc:
        raise ErroDados(f"Não foi possível abrir o CSV de terrenos: {exc.strerror or type(exc).__name__}.") from exc
    if not terrenos:
        raise ErroDados("O CSV de terrenos está vazio.")
    return terrenos


def buscar_registros(
    bairro: str = "", area_min_m2: float = 0, area_max_m2: float = 0,
    preco_max_brl: float = 0, fonte: str = "", caminho: Path = ARQUIVO_CSV,
) -> dict:
    """Seleciona terrenos por critérios objetivos e devolve códigos e fontes."""
    if min(area_min_m2, area_max_m2, preco_max_brl) < 0:
        raise ErroDados("Filtros numéricos não podem ser negativos.")
    if area_max_m2 and area_min_m2 and area_min_m2 > area_max_m2:
        raise ErroDados("A área mínima não pode ser maior que a máxima.")
    registros = carregar_terrenos(caminho)
    encontrados = [
        terreno for terreno in registros
        if (not bairro or _normalizar(bairro) in _normalizar(terreno["bairro"]))
        and (not fonte or _normalizar(fonte) in _normalizar(terreno["fonte"]))
        and (not area_min_m2 or terreno["area_m2"] >= area_min_m2)
        and (not area_max_m2 or terreno["area_m2"] <= area_max_m2)
        and (not preco_max_brl or terreno["preco_brl"] <= preco_max_brl)
    ]
    encontrados.sort(key=lambda t: t["preco_m2_brl"])
    return {
        "status": "ok" if encontrados else "sem_resultados",
        "total": len(encontrados),
        "filtros": {"bairro": bairro, "fonte": fonte, "area_min_m2": area_min_m2,
                    "area_max_m2": area_max_m2, "preco_max_brl": preco_max_brl},
        "terrenos": encontrados,
    }


def analisar_registros(codigos: list[str], caminho: Path = ARQUIVO_CSV) -> dict:
    """Calcula estatísticas exatas da seleção de códigos de anúncios.

    Os códigos vêm da primeira ferramenta. Sem códigos, não há inferência
    sobre qual conjunto deve ser analisado.
    """
    if not isinstance(codigos, list) or not codigos or not all(isinstance(c, str) and c.strip() for c in codigos):
        raise ErroDados("Informe uma lista não vazia de códigos retornados por buscar_terrenos.")
    codigos_unicos = list(dict.fromkeys(c.strip().casefold() for c in codigos))
    terrenos = carregar_terrenos(caminho)
    indice = {t["codigo"].casefold(): t for t in terrenos}
    desconhecidos = [codigo for codigo in codigos_unicos if codigo not in indice]
    if desconhecidos:
        raise ErroDados("Códigos não encontrados na base: " + ", ".join(desconhecidos))
    selecionados = [indice[codigo] for codigo in codigos_unicos]
    valores_m2 = [t["preco_brl"] / t["area_m2"] for t in selecionados]
    valores_totais = [t["preco_brl"] for t in selecionados]
    grupos = {}
    for t in selecionados:
        grupos.setdefault(t["bairro"], []).append(t["preco_brl"] / t["area_m2"])
    resumo_por_bairro = [
        {"bairro": bairro, "quantidade": len(valores),
         "mediana_preco_m2_brl": round(statistics.median(valores), 2)}
        for bairro, valores in sorted(grupos.items())
    ]
    return {
        "status": "ok",
        "quantidade": len(selecionados),
        "codigos_analisados": [t["codigo"] for t in selecionados],
        "media_preco_m2_brl": round(statistics.mean(valores_m2), 2),
        "mediana_preco_m2_brl": round(statistics.median(valores_m2), 2),
        "min_preco_m2_brl": round(min(valores_m2), 2),
        "max_preco_m2_brl": round(max(valores_m2), 2),
        "media_preco_total_brl": round(statistics.mean(valores_totais), 2),
        "mediana_preco_total_brl": round(statistics.median(valores_totais), 2),
        "quantidade_por_fonte": dict(Counter(t["fonte"] for t in selecionados)),
        "resumo_por_bairro": resumo_por_bairro,
        "observacao": "Amostra de preços anunciados, não de transações realizadas.",
    }


def criar_ferramentas():
    """Conecta as DUAS funções ao LangChain por meio de @tool."""
    from langchain.tools import tool

    @tool
    def buscar_terrenos(
        bairro: str = "", area_min_m2: float = 0, area_max_m2: float = 0,
        preco_max_brl: float = 0, fonte: str = "",
    ) -> str:
        """Pesquisa anúncios reais do CSV de terrenos de Joaçaba e retorna códigos.

        Use para descobrir terrenos, filtrar bairro, imobiliária, área e preço.
        Para obter estatísticas, passe os códigos retornados a analisar_precos.
        Sem filtros retorna TODOS os registros da base. Não consulta internet.
        """
        try:
            resultado = buscar_registros(bairro, area_min_m2, area_max_m2, preco_max_brl, fonte)
        except (ErroDados, ValueError, TypeError) as exc:
            resultado = {"status": "erro", "mensagem": str(exc)}
        return json.dumps(resultado, ensure_ascii=False)

    @tool
    def analisar_precos(codigos: list[str]) -> str:
        """Calcula estatísticas dos anúncios cujos códigos vieram de buscar_terrenos.

        Use para média, mediana, menor/maior R$/m², comparação por bairro e
        contagem por imobiliária. Exige lista de códigos devolvidos pela busca.
        Não calcule os indicadores mentalmente; use esta ferramenta Python.
        """
        try:
            resultado = analisar_registros(codigos)
        except (ErroDados, ValueError, TypeError) as exc:
            resultado = {"status": "erro", "mensagem": str(exc)}
        return json.dumps(resultado, ensure_ascii=False)

    return [buscar_terrenos, analisar_precos]
