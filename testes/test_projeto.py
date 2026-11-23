"""Testes automáticos sem consumo de tokens ou chave de API."""

import csv
import json
import statistics

import pytest

from ferramentas import ErroDados, analisar_registros, buscar_registros, carregar_terrenos


def test_base_inicial_com_26_terrenos():
    registros = carregar_terrenos()
    assert len(registros) == 26
    assert len({t["url"] for t in registros}) == 26


def test_tres_fontes_na_base():
    fontes = {t["fonte"] for t in carregar_terrenos()}
    assert fontes == {"SOMAR Imobiliária", "Pedro Fiório Imóveis", "Ivam Rover Imóveis"}


def test_precos_e_areas_validos():
    assert all(t["area_m2"] > 0 and t["preco_brl"] > 0 for t in carregar_terrenos())


def test_busca_sem_filtro_retorna_26():
    resultado = buscar_registros()
    assert resultado["status"] == "ok"
    assert resultado["total"] == 26


def test_filtro_por_bairro_sem_diferenciar_acentos():
    resultado = buscar_registros(bairro="cruzeiro do SUL")
    assert resultado["total"] == 1
    assert resultado["terrenos"][0]["codigo"] == "IR-452"


def test_busca_por_area():
    resultado = buscar_registros(area_min_m2=350, area_max_m2=450)
    assert resultado["total"] > 0
    assert all(350 <= t["area_m2"] <= 450 for t in resultado["terrenos"])


def test_filtro_por_fonte():
    resultado = buscar_registros(fonte="Ivam rover")
    assert resultado["total"] == 3


def test_filtro_por_preco_maximo():
    resultado = buscar_registros(preco_max_brl=150_000)
    assert resultado["total"] > 0
    assert all(t["preco_brl"] <= 150_000 for t in resultado["terrenos"])


def test_busca_sem_resultado():
    resultado = buscar_registros(bairro="Bairro Inexistente XYZ")
    assert resultado["status"] == "sem_resultados"
    assert resultado["terrenos"] == []


def test_filtro_invalido():
    with pytest.raises(ErroDados, match="mínima"):
        buscar_registros(area_min_m2=600, area_max_m2=300)
    with pytest.raises(ErroDados, match="negativos"):
        buscar_registros(preco_max_brl=-1)


def test_analise_calcula_preco_m2_com_precisao():
    resultado = analisar_registros(["TE173"])
    esperado = round(115000 / 360, 2)
    assert resultado["media_preco_m2_brl"] == esperado
    assert resultado["mediana_preco_m2_brl"] == esperado
    assert resultado["quantidade"] == 1


def test_mediana_geral_calculada_em_python():
    terrenos = carregar_terrenos()
    codigos = [t["codigo"] for t in terrenos]
    esperado = round(statistics.median([t["preco_brl"] / t["area_m2"] for t in terrenos]), 2)
    calculado = analisar_registros(codigos)
    assert calculado["quantidade"] == 26
    assert calculado["mediana_preco_m2_brl"] == esperado
    assert len(calculado["quantidade_por_fonte"]) == 3


def test_analise_copia_de_codigo_nao_conta_duplicado():
    assert analisar_registros(["TE173", "TE173"])["quantidade"] == 1


def test_analise_exige_codigos_validos():
    with pytest.raises(ErroDados, match="lista não vazia"):
        analisar_registros([])
    with pytest.raises(ErroDados, match="não encontrados"):
        analisar_registros(["CODIGO_INEXISTENTE"])


def test_erro_arquivo_ausente(tmp_path):
    with pytest.raises(ErroDados, match="Não foi possível abrir"):
        carregar_terrenos(tmp_path / "nao_existe.csv")


def test_erro_colunas_ausentes(tmp_path):
    arquivo = tmp_path / "ruim.csv"
    arquivo.write_text("codigo,bairro\nTE1,Centro\n", encoding="utf-8")
    with pytest.raises(ErroDados, match="colunas obrigatórias"):
        carregar_terrenos(arquivo)


def test_erro_anuncio_duplicado(tmp_path):
    dados = carregar_terrenos()
    arquivo = tmp_path / "duplicado.csv"
    colunas = ["codigo", "bairro", "cidade", "area_m2", "preco_brl", "url", "fonte"]
    with arquivo.open("w", encoding="utf-8", newline="") as saida:
        escritor = csv.DictWriter(saida, fieldnames=colunas)
        escritor.writeheader()
        for _ in range(2):
            escritor.writerow({campo: dados[0][campo] for campo in colunas})
    with pytest.raises(ErroDados, match="duplicada"):
        carregar_terrenos(arquivo)


def test_tools_langchain_sem_api():
    """Verifica as duas @tools quando LangChain está instalado."""
    pytest.importorskip("langchain")
    from ferramentas import criar_ferramentas

    ferramentas = criar_ferramentas()
    assert [f.name for f in ferramentas] == ["buscar_terrenos", "analisar_precos"]
    consulta = json.loads(ferramentas[0].invoke({"bairro": "Jardim Cidade Alta"}))
    assert consulta["total"] >= 1
    codigos = [t["codigo"] for t in consulta["terrenos"]]
    analise = json.loads(ferramentas[1].invoke({"codigos": codigos}))
    assert analise["quantidade"] == consulta["total"]
