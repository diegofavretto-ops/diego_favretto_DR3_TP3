# GeoValor — DR3/TP3: Agente inteligente para consulta de terrenos

**Aluno:** Diego Favretto  
**Disciplina:** Desenvolvimento de Data-Driven Apps com Python — DR3/TP3  
**Tecnologias:** Python, LangChain, LangGraph, Qwen (GroqCloud), Streamlit e CSV.

## 1. Problema, objetivo e público-alvo

Pesquisar manualmente anúncios de terrenos exige abrir várias páginas, filtrar bairro e área, calcular o preço por metro quadrado e comparar os valores. Isso toma tempo e aumenta o risco de erros de cálculo.

O **GeoValor** é um agente de IA que responde em português a perguntas sobre uma **amostra estática de 26 anúncios públicos de terrenos de Joaçaba/SC**. O agente decide quando consultar os registros e quando realizar uma análise estatística, mantendo o histórico da conversa. O público-alvo são estudantes, profissionais de engenharia, pesquisa imobiliária e planejamento urbano.

**Funcionalidades:** consulta por bairro, imobiliária, área e preço máximo; cálculo de média/mediana e valores mínimo/máximo de R$/m²; comparação entre bairros; memória entre perguntas; interface Streamlit; visualização das ferramentas acionadas.

**Limite do projeto:** não realiza scraping, não consulta preços em tempo real e não gera mapa de calor. Essa simplificação mantém o foco nos critérios do TP3 (agentes, ferramentas, ReAct e memória).

## 2. Base de dados e fontes

Arquivo: `dados/terrenos.csv`, com **26 anúncios** reunidos anteriormente a partir das páginas públicas de:

- [SOMAR Imobiliária](https://somar.vc/terrenos-a-venda-joacaba)
- [Pedro Fiório Imóveis](https://pedrofiorioimoveis.com.br/)
- [Ivam Rover Imóveis](https://ivamroverimoveis.com.br/)

Cada linha contém **código, bairro, município, área (m²), preço anunciado (R$), URL de origem, imobiliária e data da consulta registrada na planilha**. As datas constantes no CSV são de novembro de 2026. A base é fechada e os anúncios podem ter sido alterados ou retirados dos sites após a consulta.

**Cuidados com os dados:** são preços de **oferta**, não preços efetivamente transacionados. A amostra é pequena e não probabilística; pode haver anúncios do mesmo imóvel em diferentes imobiliárias, não identificados apenas pelos códigos. Medianas por bairro com poucos registros não representam estimativas de mercado. **O sistema não emite laudos de avaliação e não substitui a ABNT NBR 14653.**

## 3. Como o agente atende aos requisitos

| Exigência | Implementação |
|---|---|
| Problema, objetivo e casos de uso | Consulta e análise de terrenos de Joaçaba (seções 1 e 6) |
| LangChain e LLM | `create_agent` + `ChatGroq` com `qwen/qwen3.8-27b` |
| Duas ferramentas externas à LLM | `buscar_terrenos` e `analisar_precos`, decoradas com `@tool` |
| ReAct | O agente escolhe e executa ferramentas; observa resultados antes de responder |
| Memória | `InMemorySaver` e `thread_id` por conversa |
| Lógica funcional | `agente.py` conecta modelo, ferramentas, prompt e memória |
| Interface | `app.py`, Streamlit com chat, registros e painel de ações/observações |
| Testes | `testes/test_projeto.py` + roteiro de teste manual abaixo |
| Documentação e entrega | Este README e instruções de GitHub/ZIP |
| Boas práticas | Código modular, docstrings, validação de CSV, tratamento de falhas, `.gitignore` |

### Duas ferramentas

**`buscar_terrenos(bairro, area_min_m2, area_max_m2, preco_max_brl, fonte)`**: lê o CSV e devolve uma lista de anúncios encontrados com códigos, valores, bairros e URLs. Se não houver filtros, consulta os 26 registros. Os filtros são opcionais.

**`analisar_precos(codigos)`**: recebe **os códigos devolvidos pela primeira ferramenta**, acessa o CSV e calcula, com funções Python (`statistics`), a quantidade da seleção, média, mediana, valores mínimo/máximo em R$/m² e mediana por bairro. Não depende de contas inventadas pela LLM.

As duas ferramentas são funções Python externas ao modelo, registradas com o decorador `@tool` do LangChain. Não há consulta HTTP nem serviço externo além da própria API da LLM. Isso cumpre o uso de duas *tools*; se a interpretação da rubrica exigir especificamente duas APIs externas, esta versão precisaria ser ampliada.

### ReAct, memória e segurança

O agente é criado em `agente.py` por `create_agent`. Quando o usuário pede, por exemplo, “mostre os terrenos do Jardim Cidade Alta e calcule a mediana”, a sequência **esperada** é: chamar `buscar_terrenos`, observar os códigos, chamar `analisar_precos`, observar o resultado numérico e preparar a resposta. O Streamlit exibe as **ações e observações** (não o raciocínio privado da LLM).

O `InMemorySaver` mantém o histórico em RAM, separado pelo identificador `thread_id`. Uma pergunta posterior como “qual deles tem o menor R$/m²?” pode reutilizar o bairro anterior. O botão **Nova conversa** gera um novo identificador; encerrar o processo elimina a memória temporária.

A chave `GROQ_API_KEY` deve ficar **somente no arquivo `.env` local**, protegido pelo `.gitignore`. Nunca publique chaves ou tokens no GitHub. O CSV não contém dados pessoais privados; contém URLs de anúncios públicos.

## 4. Estrutura dos arquivos (9 arquivos)

```text
diego_favretto_DR3_TP3/
├── app.py                    # interface Streamlit
├── agente.py                 # LangChain + Groq + ReAct + memória
├── ferramentas.py            # busca, análise estatística e duas @tools
├── dados/
│   └── terrenos.csv          # amostra inicial de 26 anúncios
├── testes/
│   └── test_projeto.py       # testes automatizados sem API
├── requirements.txt          # dependências
├── .env.example              # exemplo de chave, sem segredo
├── .gitignore                # evita publicar credenciais/cache
└── README.md                 # documentação e guia de perguntas
```

## 5. Instalação e execução (Windows/PowerShell)

**Requisitos:** Python instalado, conexão à internet para instalar pacotes e chave GroqCloud. O plano gratuito tem limites de uso e modelos disponíveis que podem variar.

Abra esta pasta no VS Code e execute **um comando de cada vez** no terminal:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

Obtenha sua chave em <https://console.groq.com/keys>. Em `.env`, substitua a variável `GROQ_API_KEY` pela sua chave real e mantenha `GROQ_MODEL=qwen/qwen3.8-27b`. Salve. **Não envie `.env` a ninguém.**

Execute os testes e depois inicie a interface:

```powershell
.\.venv\Scripts\python.exe -m pytest -ra
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Abra o endereço informado no terminal (em geral <http://localhost:8501>). Para encerrar o servidor, use `Ctrl+C`. Nas próximas vezes, basta executar o último comando.

## 6. Perguntas sugeridas e casos de teste

| Teste | Perguntas no chat, uma de cada vez | Critério esperado |
|---|---|---|
| Busca simples | “Quantos terrenos estão na base e quais são as imobiliárias?” | `buscar_terrenos`; 26 registros no CSV inicial |
| Filtro | “Mostre os terrenos do Jardim Cidade Alta.” | Retorna códigos, preço, área e fonte somente do bairro |
| Análise | “Qual é a mediana do preço por m² dos terrenos do Jardim Cidade Alta?” | Usa `buscar_terrenos` **e** `analisar_precos` |
| Comparação | “Compare as medianas de preço/m² dos bairros disponíveis.” | Usa as duas ferramentas, informa amostra por bairro |
| Memória | “Mostre os terrenos do Jardim Cidade Alta.” Depois: “Qual deles é o mais barato por m²?” | Mantém o contexto na mesma conversa |
| Nova conversa | Clique “Nova conversa”; pergunte “Qual bairro eu estava analisando?” | Não conhece o bairro da conversa anterior |
| Sem resultados | “Encontre terrenos no bairro Inexistente XYZ.” | Resposta sem inventar anúncios |
| Sem scraping | “Atualize os anúncios das imobiliárias.” | Explica que a base é estática e não faz scraping |

**Dica para o teste de ReAct:** abra **Ações e observações (ReAct)** abaixo da resposta. Tire uma captura mostrando a ferramenta `buscar_terrenos` e outra com `analisar_precos`. Isso comprova as duas *tools* em uso.

## 7. Testes e resultados (preencher com execuções reais)

Os testes automatizados são executados **sem chave de API** via pytest. Validam: integridade do CSV, número de registros, filtros, normalização de texto, médias/mediana, erros e chamadas diretas das `@tools` (quando LangChain está instalado).

**Resultado da verificação offline neste ambiente:** **17 testes aprovados, 1 teste ignorado** (as `@tools` exigem LangChain, não instalado neste ambiente), **0 falhas**. Esse resultado não comprova uma chamada real à API Groq.

**Resultados que devem ser registrados após executar no seu PC:**

- `pytest -ra`:collected 18 items                                                                                                                           

testes\test_projeto.py ..................                                                                                              [100%]

============================== 18 passed, 17 warnings in 0.78s ======================================================                                                                                       [100%]

- Teste de ReAct com Groq e ambas as ferramentas: Aprovado.
- Teste de memória na mesma conversa: Aprovado.
- Nova conversa com `thread_id` distinto: Aprovado.


## 8. Por que o agente ajuda, comparado ao processo manual?

**Sem o agente:** o pesquisador abriria uma planilha, filtraria os imóveis por bairro, identificaria códigos, calcularia `preço ÷ área` para cada anúncio, executaria média/mediana e montaria um resumo.

**Com o agente:** a pessoa formula perguntas em linguagem natural; a LLM escolhe as ferramentas e obtém os resultados de funções Python determinísticas. A memória reduz repetição de filtros entre perguntas. Isso melhora a experiência de consulta, mas não elimina a conferência dos dados ou a validação técnica.

## 9. Referências e entrega

- [Documentação oficial dos agentes LangChain](https://docs.langchain.com/oss/python/langchain/agents)
- [Documentação oficial das tools](https://docs.langchain.com/oss/python/langchain/tools)
- [Memória de agentes LangChain](https://docs.langchain.com/oss/python/langchain/short-term-memory)
- [Modelos disponíveis no Groq](https://console.groq.com/docs/models)


