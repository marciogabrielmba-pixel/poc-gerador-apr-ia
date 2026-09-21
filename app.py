import streamlit as st
import zipfile
import io
import re
import time
import unicodedata
from collections import defaultdict
from openpyxl import load_workbook

# ============================================================
# CONFIGURAÇÃO
# ============================================================

st.set_page_config(
    page_title="Gerador de APR com IA",
    page_icon="🦺",
    layout="wide"
)

st.title("🦺 Gerador de APR com IA")
st.subheader("POC — Consulta inteligente à base de APRs")

st.info(
    "Objetivo do POC: consultar uma base real de APRs, "
    "encontrar atividades semelhantes e preparar a base "
    "para geração automática de novas APRs."
)

# ============================================================
# INDICADORES
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🎯 Meta", "10 APRs")

with col2:
    st.metric("⏱️ Meta de entrega", "50 minutos")

with col3:
    st.metric("⚡ Meta por APR", "≤ 5 minutos")

st.divider()

# ============================================================
# STOPWORDS
# ============================================================

STOPWORDS = {
    "a", "o", "e", "de", "da", "do", "das", "dos",
    "em", "no", "na", "nos", "nas", "para", "por",
    "com", "sem", "um", "uma", "uns", "umas",
    "ao", "aos", "as", "os", "que", "se", "é",
    "ser", "foi", "são", "como", "ou", "não",
    "mais", "menos", "entre", "sobre", "pela",
    "pelo", "pelas", "pelos"
}

# ============================================================
# NORMALIZAÇÃO
# ============================================================

def normalizar(texto):
    texto = str(texto)

    texto = unicodedata.normalize(
        "NFKD",
        texto
    ).encode(
        "ASCII",
        "ignore"
    ).decode(
        "ASCII"
    )

    texto = texto.lower()

    texto = re.sub(
        r"[^a-z0-9\s]",
        " ",
        texto
    )

    texto = re.sub(
        r"\s+",
        " ",
        texto
    ).strip()

    return texto


def tokens(texto):
    texto = normalizar(texto)

    return [
        palavra
        for palavra in texto.split()
        if len(palavra) > 2
        and palavra not in STOPWORDS
    ]


def tokens_unicos(texto):
    return set(tokens(texto))


# ============================================================
# SCORE DE RELEVÂNCIA
# ============================================================

def calcular_score(consulta, linha):
    """
    Calcula relevância da consulta em relação a uma linha.
    
    Critérios:
    - cobertura dos termos;
    - frequência;
    - proximidade;
    - correspondência da expressão;
    """

    consulta_normalizada = normalizar(consulta)
    linha_normalizada = normalizar(linha)

    consulta_tokens = tokens_unicos(consulta)
    linha_tokens = tokens(linha)

    if not consulta_tokens or not linha_tokens:
        return 0

    # --------------------------------------------------------
    # 1. Cobertura dos termos
    # --------------------------------------------------------

    encontrados = consulta_tokens.intersection(
        set(linha_tokens)
    )

    cobertura = (
        len(encontrados) /
        len(consulta_tokens)
    )

    # --------------------------------------------------------
    # 2. Frequência
    # --------------------------------------------------------

    frequencia = 0

    for termo in encontrados:

        frequencia += linha_tokens.count(termo)

    frequencia_bonus = min(
        frequencia * 2,
        10
    )

    # --------------------------------------------------------
    # 3. Expressão exata
    # --------------------------------------------------------

    frase_bonus = 0

    if (
        len(consulta_normalizada) > 5
        and consulta_normalizada in linha_normalizada
    ):
        frase_bonus = 25

    # --------------------------------------------------------
    # 4. Proximidade dos termos
    # --------------------------------------------------------

    proximidade_bonus = 0

    posicoes = []

    for termo in consulta_tokens:

        for i, palavra in enumerate(linha_tokens):

            if palavra == termo:

                posicoes.append(i)

                break

    if len(posicoes) >= 2:

        distancia = (
            max(posicoes) -
            min(posicoes)
        )

        if distancia <= 5:
            proximidade_bonus = 15

        elif distancia <= 10:
            proximidade_bonus = 8

    # --------------------------------------------------------
    # 5. Termos principais
    # --------------------------------------------------------

    termo_bonus = 0

    for termo in encontrados:

        if len(termo) >= 8:
            termo_bonus += 3

    termo_bonus = min(
        termo_bonus,
        15
    )

    # --------------------------------------------------------
    # SCORE FINAL
    # --------------------------------------------------------

    score = (
        cobertura * 55
        + frequencia_bonus
        + frase_bonus
        + proximidade_bonus
        + termo_bonus
    )

    return min(
        score,
        100
    )


# ============================================================
# LEITURA DAS PLANILHAS
# ============================================================

def ler_excel(conteudo, nome_arquivo):

    registros = []

    try:

        arquivo = io.BytesIO(conteudo)

        workbook = load_workbook(
            arquivo,
            read_only=True,
            data_only=True
        )

        for nome_planilha in workbook.sheetnames:

            planilha = workbook[
                nome_planilha
            ]

            numero_linha = 0

            for linha in planilha.iter_rows(
                values_only=True
            ):

                numero_linha += 1

                valores = []

                for valor in linha:

                    if valor is not None:

                        texto = str(valor).strip()

                        if texto:

                            valores.append(texto)

                if not valores:
                    continue

                texto_linha = " | ".join(
                    valores
                )

                registros.append({
                    "arquivo": nome_arquivo,
                    "planilha": nome_planilha,
                    "linha": numero_linha,
                    "texto": texto_linha
                })

        workbook.close()

    except Exception as erro:

        registros.append({
            "arquivo": nome_arquivo,
            "planilha": "ERRO",
            "linha": 0,
            "texto": f"Erro: {erro}"
        })

    return registros


# ============================================================
# CONSOLIDAR RESULTADOS POR APR
# ============================================================

def consolidar_resultados(
    resultados,
    quantidade
):

    grupos = defaultdict(list)

    for resultado in resultados:

        grupos[
            resultado["arquivo"]
        ].append(resultado)

    aprs = []

    for arquivo, linhas in grupos.items():

        linhas.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        melhor_linha = linhas[0]

        # Penaliza ligeiramente resultados
        # baseados em apenas uma ocorrência
        ocorrencias = len(linhas)

        bonus_ocorrencia = min(
            ocorrencias * 1.5,
            8
        )

        score_final = min(
            melhor_linha["score"]
            + bonus_ocorrencia,
            100
        )

        aprs.append({
            "arquivo": arquivo,
            "planilha": melhor_linha["planilha"],
            "linha": melhor_linha["linha"],
            "score": score_final,
            "melhor_trecho": melhor_linha["texto"],
            "ocorrencias": ocorrencias
        })

    aprs.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return aprs[:quantidade]


# ============================================================
# UPLOAD DA BASE
# ============================================================

st.header("📦 1. Carregar base FONTE")

st.write(
    "Envie o arquivo **FONTE.zip** contendo as APRs em Excel."
)

arquivo_zip = st.file_uploader(
    "Selecione o FONTE.zip",
    type=["zip"]
)

if arquivo_zip:

    tamanho_mb = (
        arquivo_zip.size /
        (1024 * 1024)
    )

    st.success(
        f"Arquivo recebido: **{arquivo_zip.name}** "
        f"({tamanho_mb:.1f} MB)"
    )

    if st.button(
        "🔎 INDEXAR BASE DE APRs",
        type="primary"
    ):

        inicio = time.time()

        registros_base = []

        progresso = st.progress(0)

        status = st.empty()

        try:

            arquivo_zip.seek(0)

            with zipfile.ZipFile(
                arquivo_zip,
                "r"
            ) as zip_ref:

                arquivos_excel = [
                    nome
                    for nome in zip_ref.namelist()
                    if nome.lower().endswith(
                        (".xlsx", ".xlsm")
                    )
                    and not nome.startswith(
                        "__MACOSX"
                    )
                ]

                total = len(
                    arquivos_excel
                )

                if total == 0:

                    st.error(
                        "Nenhum arquivo Excel (.xlsx ou .xlsm) "
                        "foi encontrado dentro do ZIP."
                    )

                else:

                    for contador, nome_arquivo in enumerate(
                        arquivos_excel,
                        start=1
                    ):

                        status.text(
                            f"Lendo {contador} de {total}: "
                            f"{nome_arquivo}"
                        )

                        conteudo = zip_ref.read(
                            nome_arquivo
                        )

                        dados = ler_excel(
                            conteudo,
                            nome_arquivo
                        )

                        registros_base.extend(
                            dados
                        )

                        progresso.progress(
                            contador / total
                        )

            tempo = (
                time.time() -
                inicio
            )

            st.session_state[
                "registros_base"
            ] = registros_base

            st.session_state[
                "quantidade_arquivos"
            ] = total

            st.session_state[
                "tempo_indexacao"
            ] = tempo

            status.empty()

            st.success(
                f"✅ Base indexada com sucesso em "
                f"**{tempo:.1f} segundos**."
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Arquivos Excel",
                    total
                )

            with col2:

                st.metric(
                    "Linhas indexadas",
                    len(registros_base)
                )

            with col3:

                st.metric(
                    "Tempo",
                    f"{tempo:.1f}s"
                )

        except Exception as erro:

            st.error(
                f"Erro durante a indexação: {erro}"
            )


# ============================================================
# PESQUISA
# ============================================================

if "registros_base" in st.session_state:

    st.divider()

    st.header(
        "🔎 2. Pesquisar na base de APRs"
    )

    atividade = st.text_input(
        "Digite a atividade para pesquisar:",
        placeholder=(
            "Ex.: Instalação de eletrocalhas em altura"
        )
    )

    quantidade_resultados = st.slider(
        "Quantidade de APRs semelhantes:",
        min_value=1,
        max_value=10,
        value=5
    )

    if st.button(
        "🔍 PESQUISAR APRs",
        type="primary"
    ):

        if not atividade.strip():

            st.warning(
                "Digite uma atividade para realizar a pesquisa."
            )

        else:

            inicio_pesquisa = time.time()

            resultados_linhas = []

            for registro in st.session_state[
                "registros_base"
            ]:

                score = calcular_score(
                    atividade,
                    registro["texto"]
                )

                if score > 0:

                    resultados_linhas.append({
                        "score": score,
                        "arquivo": registro["arquivo"],
                        "planilha": registro["planilha"],
                        "linha": registro["linha"],
                        "texto": registro["texto"]
                    })

            resultados = consolidar_resultados(
                resultados_linhas,
                quantidade_resultados
            )

            tempo_pesquisa = (
                time.time()
                - inicio_pesquisa
            )

            st.success(
                f"Pesquisa concluída em "
                f"**{tempo_pesquisa:.3f} segundos**."
            )

            if not resultados:

                st.warning(
                    "Nenhuma APR semelhante foi encontrada."
                )

            else:

                st.subheader(
                    f"📋 {len(resultados)} APR(s) encontrada(s)"
                )

                for numero, resultado in enumerate(
                    resultados,
                    start=1
                ):

                    with st.expander(
                        f"#{numero} — "
                        f"{resultado['arquivo']} "
                        f"— Relevância "
                        f"{resultado['score']:.1f}%"
                    ):

                        st.write(
                            f"**Arquivo:** "
                            f"{resultado['arquivo']}"
                        )

                        st.write(
                            f"**Planilha:** "
                            f"{resultado['planilha']}"
                        )

                        st.write(
                            f"**Linha mais relevante:** "
                            f"{resultado['linha']}"
                        )

                        st.write(
                            f"**Relevância calculada:** "
                            f"{resultado['score']:.1f}%"
                        )

                        st.write(
                            f"**Ocorrências relevantes:** "
                            f"{resultado['ocorrencias']}"
                        )

                        st.write(
                            "**Trecho que gerou a relevância:**"
                        )

                        st.info(
                            resultado["melhor_trecho"]
                        )


# ============================================================
# PRÓXIMA ETAPA
# ============================================================

st.divider()

st.header(
    "🚀 Próxima etapa do POC"
)

st.write(
    """
Depois de validarmos a recuperação das APRs reais,
vamos evoluir o sistema para:

1. Busca semântica com IA;
2. Identificação automática de perigos;
3. Identificação automática de riscos;
4. Identificação de consequências;
5. Identificação de controles;
6. Consulta aos requisitos legais aplicáveis;
7. Geração automática da nova APR;
8. Preservação do layout padrão da empresa;
9. Exportação para Excel;
10. Medição da meta de **≤ 5 minutos por APR**.
"""
)

st.caption(
    "V4 — Motor de recuperação por relevância de linhas. "
    "O percentual apresentado é um índice de relevância do POC, "
    "não uma probabilidade ou garantia de similaridade técnica."
)
