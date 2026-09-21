import streamlit as st
import zipfile
import io
import re
import time
import numpy as np
from collections import Counter
from openpyxl import load_workbook
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURAÇÃO
# ============================================================

st.set_page_config(
    page_title="Gerador de APR com IA",
    page_icon="🦺",
    layout="wide"
)


# ============================================================
# TÍTULO
# ============================================================

st.title("🦺 Gerador de APR com IA")

st.subheader("POC — Consulta contextual à base de APRs")

st.info(
    "V5.1 — Ranking técnico inteligente: combina "
    "similaridade semântica, relevância técnica e "
    "raridade dos termos na base."
)


# ============================================================
# METAS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.caption("🎯 Meta")
    st.metric("APRs", "10 APRs")

with col2:
    st.caption("🎯 Meta")
    st.metric("Tempo total", "50 minutos")

with col3:
    st.caption("⚡ Meta por APR")
    st.metric("Tempo", "≤ 5 minutos")


st.divider()


# ============================================================
# STOPWORDS
# ============================================================

STOPWORDS = {
    "a", "o", "as", "os",
    "um", "uma", "uns", "umas",
    "de", "da", "do", "das", "dos",
    "em", "no", "na", "nos", "nas",
    "por", "para",
    "com", "sem",
    "e", "ou",
    "que", "se",
    "ao", "aos",
    "à", "às",
    "é", "ser",
    "como",
    "mais", "menos",
    "sobre",
    "entre",
    "durante",
    "após",
    "antes",
    "até",
    "pelo", "pela",
    "pelos", "pelas"
}


# ============================================================
# TERMOS GENÉRICOS
# ============================================================

TERMOS_GENERICOS = {
    "instalacao",
    "montagem",
    "servico",
    "execucao",
    "atividade",
    "trabalho",
    "sistema",
    "processo",
    "realizacao",
    "manutencao",
    "operacao",
    "estrutura",
    "equipamento",
    "procedimento",
    "local",
    "material",
    "equipe",
    "fixacao"
}


# ============================================================
# NORMALIZAÇÃO
# ============================================================

def normalizar(texto):

    texto = str(texto).lower()

    substituicoes = {
        "á": "a",
        "à": "a",
        "ã": "a",
        "â": "a",
        "ä": "a",

        "é": "e",
        "è": "e",
        "ê": "e",
        "ë": "e",

        "í": "i",
        "ì": "i",
        "î": "i",
        "ï": "i",

        "ó": "o",
        "ò": "o",
        "õ": "o",
        "ô": "o",
        "ö": "o",

        "ú": "u",
        "ù": "u",
        "û": "u",
        "ü": "u",

        "ç": "c"
    }

    for origem, destino in substituicoes.items():
        texto = texto.replace(origem, destino)

    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto)

    return texto.strip()


# ============================================================
# VARIANTES SINGULAR / PLURAL
# ============================================================

def variantes_termo(termo):

    variantes = {termo}

    # Ex.: eletrocalhas -> eletrocalha
    if termo.endswith("s") and len(termo) > 4:
        variantes.add(termo[:-1])

    # Ex.: instalações -> instalacao
    if termo.endswith("coes") and len(termo) > 6:
        variantes.add(
            termo[:-4] + "cao"
        )

    return variantes


# ============================================================
# TOKENIZAÇÃO
# ============================================================

def obter_tokens(texto):

    texto = normalizar(texto)

    tokens = texto.split()

    tokens = [
        token
        for token in tokens
        if token not in STOPWORDS
        and len(token) > 2
    ]

    return tokens


# ============================================================
# TERMOS TÉCNICOS
# ============================================================

def peso_base_termo(termo):

    if termo in TERMOS_GENERICOS:
        return 0.20

    return 1.0


# ============================================================
# ÍNDICE DE RARIDADE
# ============================================================

def calcular_idf(df, total_documentos):

    if total_documentos <= 0:
        return 1.0

    return np.log(
        (total_documentos + 1)
        /
        (df + 1)
    ) + 1


# ============================================================
# CRIAR ESTATÍSTICAS DA BASE
# ============================================================

def criar_estatisticas_base(base_aprs):

    documentos = {}

    for item in base_aprs:

        arquivo = item["arquivo"]

        if arquivo not in documentos:
            documentos[arquivo] = set()

        tokens = obter_tokens(
            item["texto"]
        )

        for token in tokens:

            for variante in variantes_termo(token):
                documentos[arquivo].add(variante)


    df = Counter()

    for tokens in documentos.values():

        for token in tokens:
            df[token] += 1


    total_documentos = len(
        documentos
    )


    idf = {}

    for termo, frequencia in df.items():

        idf[termo] = calcular_idf(
            frequencia,
            total_documentos
        )


    return idf, total_documentos


# ============================================================
# SCORE TÉCNICO INTELIGENTE
# ============================================================

def calcular_score_tecnico(
    consulta,
    texto,
    idf,
    total_documentos
):

    tokens_consulta = obter_tokens(
        consulta
    )

    if not tokens_consulta:
        return 0.0, [], 0.0


    texto_normalizado = normalizar(
        texto
    )

    tokens_texto = set(
        obter_tokens(texto)
    )


    pontos = 0.0
    pontos_maximos = 0.0

    termos_encontrados = []

    pesos_termos = []


    for termo in tokens_consulta:

        variantes = variantes_termo(
            termo
        )


        # ----------------------------------------------------
        # RARIDADE
        # ----------------------------------------------------

        raridade = idf.get(
            termo,
            calcular_idf(
                total_documentos,
                total_documentos
            )
        )


        peso_base = peso_base_termo(
            termo
        )


        peso = peso_base * raridade


        # Termo técnico específico recebe bônus
        if termo not in TERMOS_GENERICOS:
            peso *= 1.35


        pesos_termos.append(
            (termo, peso)
        )

        pontos_maximos += peso


        encontrado = False


        # ----------------------------------------------------
        # VERIFICAÇÃO DO TERMO
        # ----------------------------------------------------

        for variante in variantes:

            if variante in tokens_texto:

                encontrado = True
                break


        if encontrado:

            termos_encontrados.append(
                termo
            )

            pontos += peso


    # --------------------------------------------------------
    # COBERTURA DOS TERMOS
    # --------------------------------------------------------

    if pontos_maximos > 0:

        cobertura = (
            pontos
            /
            pontos_maximos
        ) * 100

    else:

        cobertura = 0


    # --------------------------------------------------------
    # BÔNUS DE FRASE
    # --------------------------------------------------------

    consulta_normalizada = normalizar(
        consulta
    )

    # Quanto mais partes importantes da consulta aparecem
    # próximas, maior o bônus.

    tokens_importantes = [
        termo
        for termo in tokens_consulta
        if termo not in TERMOS_GENERICOS
    ]


    bonus_frase = 0


    if len(tokens_importantes) >= 2:

        sequencias_encontradas = 0

        for termo in tokens_importantes:

            variantes = variantes_termo(
                termo
            )

            if any(
                re.search(
                    r"\b" + re.escape(v) + r"\b",
                    texto_normalizado
                )
                for v in variantes
            ):
                sequencias_encontradas += 1


        if sequencias_encontradas >= 2:
            bonus_frase = 10

        if sequencias_encontradas >= 3:
            bonus_frase = 15


    score = min(
        100,
        cobertura + bonus_frase
    )


    # --------------------------------------------------------
    # TERMO PRINCIPAL
    # --------------------------------------------------------

    if tokens_importantes:

        termo_principal = max(
            tokens_importantes,
            key=lambda x: idf.get(
                x,
                1.0
            )
        )

        peso_principal = idf.get(
            termo_principal,
            1.0
        )

    else:

        peso_principal = 1.0


    return (
        score,
        termos_encontrados,
        peso_principal
    )


# ============================================================
# LEITURA DO EXCEL
# ============================================================

def ler_excel(
    arquivo_bytes,
    nome_arquivo
):

    resultados = []

    try:

        workbook = load_workbook(
            io.BytesIO(
                arquivo_bytes
            ),
            read_only=True,
            data_only=True
        )


        for nome_planilha in workbook.sheetnames:

            sheet = workbook[
                nome_planilha
            ]

            linhas = []


            for row in sheet.iter_rows(
                values_only=True
            ):

                valores = []


                for valor in row:

                    if valor is not None:

                        texto = str(
                            valor
                        ).strip()

                        if texto:
                            valores.append(
                                texto
                            )


                if valores:

                    linhas.append(
                        " | ".join(
                            valores
                        )
                    )


            texto_planilha = (
                "\n".join(
                    linhas
                ).strip()
            )


            if texto_planilha:

                resultados.append({

                    "arquivo":
                        nome_arquivo,

                    "planilha":
                        nome_planilha,

                    "texto":
                        texto_planilha

                })


        workbook.close()


    except Exception as e:

        st.warning(
            f"Erro ao ler "
            f"{nome_arquivo}: {e}"
        )


    return resultados


# ============================================================
# CHUNKS
# ============================================================

def criar_chunks(
    texto,
    tamanho=1800,
    sobreposicao=250
):

    texto = str(texto)


    if len(texto) <= tamanho:
        return [texto]


    chunks = []

    inicio = 0


    while inicio < len(texto):

        fim = (
            inicio
            +
            tamanho
        )

        chunk = texto[
            inicio:fim
        ]


        if chunk.strip():

            chunks.append(
                chunk.strip()
            )


        inicio = (
            fim
            -
            sobreposicao
        )


        if inicio >= len(texto):
            break


    return chunks


# ============================================================
# MODELO
# ============================================================

@st.cache_resource
def carregar_modelo():

    return SentenceTransformer(
        "sentence-transformers/"
        "paraphrase-multilingual-MiniLM-L12-v2"
    )


# ============================================================
# PREPARAR ÍNDICE SEMÂNTICO
# ============================================================

def preparar_indice_semantico(
    base_aprs,
    modelo
):

    chunks = []

    metadados = []


    for item in base_aprs:

        partes = criar_chunks(
            item["texto"]
        )


        for numero, parte in enumerate(
            partes
        ):

            texto_embedding = (
                f"Arquivo: "
                f"{item['arquivo']}\n"
                f"Planilha: "
                f"{item['planilha']}\n"
                f"{parte}"
            )


            chunks.append(
                texto_embedding
            )


            metadados.append({

                "arquivo":
                    item["arquivo"],

                "planilha":
                    item["planilha"],

                "texto":
                    item["texto"],

                "chunk":
                    numero

            })


    if not chunks:

        return None, []


    embeddings = modelo.encode(
        chunks,
        batch_size=16,
        show_progress_bar=False,
        normalize_embeddings=True
    )


    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )


    return (
        embeddings,
        metadados
    )


# ============================================================
# BUSCA V5.1
# ============================================================

def buscar_aprs(
    consulta,
    modelo,
    embeddings,
    metadados,
    base_aprs,
    idf,
    total_documentos,
    quantidade=10
):

    query_embedding = modelo.encode(
        [consulta],
        normalize_embeddings=True,
        show_progress_bar=False
    )[0]


    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # SIMILARIDADE SEMÂNTICA
    # --------------------------------------------------------

    similaridades = np.dot(
        embeddings,
        query_embedding
    )


    melhores_semanticos = {}


    for indice, similaridade in enumerate(
        similaridades
    ):

        arquivo = metadados[
            indice
        ]["arquivo"]


        if (
            arquivo not in
            melhores_semanticos
            or
            similaridade >
            melhores_semanticos[
                arquivo
            ]["similaridade"]
        ):

            melhores_semanticos[
                arquivo
            ] = {

                "similaridade":
                    float(
                        similaridade
                    ),

                "planilha":
                    metadados[
                        indice
                    ]["planilha"],

                "chunk":
                    metadados[
                        indice
                    ]["chunk"]

            }


    resultados = []


    for item in base_aprs:

        arquivo = item[
            "arquivo"
        ]


        if arquivo not in (
            melhores_semanticos
        ):
            continue


        similaridade = (
            melhores_semanticos[
                arquivo
            ]["similaridade"]
        )


        # ----------------------------------------------------
        # SEMÂNTICA
        # ----------------------------------------------------

        score_semantico = max(
            0,
            min(
                100,
                (
                    (
                        similaridade
                        +
                        1
                    )
                    /
                    2
                )
                *
                100
            )
        )


        # ----------------------------------------------------
        # TÉCNICO
        # ----------------------------------------------------

        (
            score_tecnico,
            termos,
            peso_principal
        ) = calcular_score_tecnico(
            consulta,
            item["texto"],
            idf,
            total_documentos
        )


        # ----------------------------------------------------
        # SCORE HÍBRIDO V5.1
        # ----------------------------------------------------

        score_hibrido = (
            score_semantico * 0.55
            +
            score_tecnico * 0.45
        )


        # ----------------------------------------------------
        # BÔNUS PARA TERMO TÉCNICO PRINCIPAL
        # ----------------------------------------------------

        tokens_consulta = obter_tokens(
            consulta
        )


        termos_especificos = [
            termo
            for termo in tokens_consulta
            if termo not in TERMOS_GENERICOS
        ]


        texto_normalizado = normalizar(
            item["texto"]
        )


        principal_encontrado = False


        if termos_especificos:

            principal = max(
                termos_especificos,
                key=lambda x: idf.get(
                    x,
                    1.0
                )
            )


            for variante in variantes_termo(
                principal
            ):

                if re.search(
                    r"\b"
                    +
                    re.escape(variante)
                    +
                    r"\b",
                    texto_normalizado
                ):

                    principal_encontrado = True
                    break


        if principal_encontrado:

            score_hibrido += 5


        score_hibrido = min(
            score_hibrido,
            100
        )


        resultados.append({

            "arquivo":
                arquivo,

            "planilha":
                item["planilha"],

            "score_semantico":
                score_semantico,

            "score_tecnico":
                score_tecnico,

            "score_hibrido":
                score_hibrido,

            "termos":
                termos,

            "texto":
                item["texto"]

        })


    # --------------------------------------------------------
    # ORDENAR
    # --------------------------------------------------------

    resultados.sort(
        key=lambda x:
            x["score_hibrido"],
        reverse=True
    )


    # --------------------------------------------------------
    # DEDUPLICAR
    # --------------------------------------------------------

    finais = []

    arquivos_vistos = set()


    for resultado in resultados:

        if (
            resultado["arquivo"]
            in arquivos_vistos
        ):
            continue


        arquivos_vistos.add(
            resultado["arquivo"]
        )


        finais.append(
            resultado
        )


        if len(finais) >= quantidade:
            break


    return finais


# ============================================================
# SESSION STATE
# ============================================================

if "base_aprs" not in st.session_state:
    st.session_state.base_aprs = []


if "indice_embeddings" not in st.session_state:
    st.session_state.indice_embeddings = None


if "metadados_embeddings" not in st.session_state:
    st.session_state.metadados_embeddings = []


if "indice_pronto" not in st.session_state:
    st.session_state.indice_pronto = False


if "arquivo_processado" not in st.session_state:
    st.session_state.arquivo_processado = None


if "idf" not in st.session_state:
    st.session_state.idf = {}


if "total_documentos" not in st.session_state:
    st.session_state.total_documentos = 0


# ============================================================
# 1. CARREGAR BASE
# ============================================================

st.header(
    "📦 1. Carregar base FONTE"
)


uploaded_file = st.file_uploader(
    "Selecione o FONTE.zip",
    type=["zip"]
)


if uploaded_file is not None:

    nome_zip = uploaded_file.name


    if (
        st.session_state.arquivo_processado
        != nome_zip
    ):

        st.session_state.base_aprs = []

        st.session_state.indice_embeddings = None

        st.session_state.metadados_embeddings = []

        st.session_state.indice_pronto = False

        st.session_state.idf = {}

        st.session_state.total_documentos = 0


    if st.button(
        "📥 INDEXAR BASE FONTE",
        type="primary"
    ):

        inicio = time.time()


        with st.spinner(
            "Lendo todos os arquivos Excel..."
        ):

            base_aprs = []


            with zipfile.ZipFile(
                uploaded_file,
                "r"
            ) as zip_ref:

                arquivos_excel = [

                    nome

                    for nome
                    in zip_ref.namelist()

                    if (
                        nome.lower().endswith(
                            ".xlsx"
                        )
                        or
                        nome.lower().endswith(
                            ".xlsm"
                        )
                    )
                    and
                    not nome.startswith(
                        "__MACOSX"
                    )

                ]


                total_arquivos = len(
                    arquivos_excel
                )


                progresso = st.progress(
                    0
                )


                for contador, nome in enumerate(
                    arquivos_excel,
                    start=1
                ):

                    try:

                        arquivo_bytes = (
                            zip_ref.read(
                                nome
                            )
                        )


                        dados = ler_excel(
                            arquivo_bytes,
                            nome
                        )


                        base_aprs.extend(
                            dados
                        )


                    except Exception as e:

                        st.warning(
                            f"Erro em "
                            f"{nome}: {e}"
                        )


                    progresso.progress(
                        contador
                        /
                        total_arquivos
                    )


        st.session_state.base_aprs = (
            base_aprs
        )


        st.session_state.arquivo_processado = (
            nome_zip
        )


        # Estatísticas para ranking técnico
        (
            idf,
            total_documentos
        ) = criar_estatisticas_base(
            base_aprs
        )


        st.session_state.idf = idf

        st.session_state.total_documentos = (
            total_documentos
        )


        st.session_state.indice_embeddings = None

        st.session_state.metadados_embeddings = []

        st.session_state.indice_pronto = False


        tempo = (
            time.time()
            -
            inicio
        )


        arquivos_unicos = len(
            set(
                item["arquivo"]
                for item in base_aprs
            )
        )


        st.success(
            f"Base indexada: "
            f"{arquivos_unicos} arquivos "
            f"e "
            f"{len(base_aprs)} planilhas "
            f"em "
            f"{tempo:.1f} segundos."
        )


# ============================================================
# STATUS
# ============================================================

if st.session_state.base_aprs:

    st.divider()

    st.header(
        "🧠 2. Preparar busca semântica"
    )


    total_aprs = len(
        set(
            item["arquivo"]
            for item
            in st.session_state.base_aprs
        )
    )


    total_planilhas = len(
        st.session_state.base_aprs
    )


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Arquivos Excel",
            total_aprs
        )


    with col2:

        st.metric(
            "Planilhas indexadas",
            total_planilhas
        )


    st.info(
        "V5.1: além da semântica, "
        "o sistema calcula a raridade dos "
        "termos técnicos dentro da própria base."
    )


    if st.button(
        "🚀 PREPARAR BUSCA SEMÂNTICA",
        type="primary"
    ):

        inicio_semantico = time.time()


        try:

            with st.spinner(
                "Carregando modelo semântico..."
            ):

                modelo = (
                    carregar_modelo()
                )


            with st.spinner(
                "Criando índice semântico..."
            ):

                (
                    embeddings,
                    metadados
                ) = preparar_indice_semantico(
                    st.session_state.base_aprs,
                    modelo
                )


            st.session_state.indice_embeddings = (
                embeddings
            )


            st.session_state.metadados_embeddings = (
                metadados
            )


            st.session_state.indice_pronto = (
                True
            )


            tempo_semantico = (
                time.time()
                -
                inicio_semantico
            )


            st.success(
                f"Busca semântica preparada "
                f"em "
                f"{tempo_semantico:.1f} segundos."
            )


        except Exception as e:

            st.error(
                "Não foi possível preparar "
                "o índice semântico."
            )

            st.exception(e)


# ============================================================
# 3. CONSULTA
# ============================================================

if st.session_state.indice_pronto:

    st.divider()


    st.header(
        "🔎 3. Consultar base de APRs"
    )


    consulta = st.text_input(
        "Descreva a atividade que você deseja encontrar:",
        placeholder=(
            "Ex.: Montagem e fixação "
            "de eletrocalhas em altura"
        )
    )


    quantidade = st.slider(
        "Quantidade de APRs semelhantes",
        min_value=3,
        max_value=20,
        value=10
    )


    if st.button(
        "🔍 BUSCAR APRs SEMANTICAMENTE",
        type="primary"
    ):

        if not consulta.strip():

            st.warning(
                "Digite uma atividade "
                "para realizar a busca."
            )


        else:

            inicio_busca = time.time()


            with st.spinner(
                "Calculando compatibilidade técnica..."
            ):

                modelo = (
                    carregar_modelo()
                )


                resultados = buscar_aprs(

                    consulta,

                    modelo,

                    st.session_state.indice_embeddings,

                    st.session_state.metadados_embeddings,

                    st.session_state.base_aprs,

                    st.session_state.idf,

                    st.session_state.total_documentos,

                    quantidade

                )


            tempo_busca = (
                time.time()
                -
                inicio_busca
            )


            st.success(
                f"Busca concluída em "
                f"{tempo_busca:.2f} segundos."
            )


            if not resultados:

                st.warning(
                    "Nenhuma APR encontrada."
                )


            else:

                st.subheader(
                    f"📋 {len(resultados)} "
                    f"APRs mais relacionadas"
                )


                st.caption(
                    "O índice híbrido é um indicador "
                    "de compatibilidade técnica para "
                    "priorização. Não representa "
                    "probabilidade de equivalência."
                )


                for posicao, resultado in enumerate(
                    resultados,
                    start=1
                ):

                    st.markdown(
                        f"### "
                        f"{posicao}. "
                        f"{resultado['arquivo']}"
                    )


                    col1, col2, col3 = (
                        st.columns(3)
                    )


                    with col1:

                        st.metric(
                            "🧠 Semântica",
                            f"{resultado['score_semantico']:.1f}%"
                        )


                    with col2:

                        st.metric(
                            "🎯 Compatibilidade técnica",
                            f"{resultado['score_tecnico']:.1f}%"
                        )


                    with col3:

                        st.metric(
                            "⭐ Índice híbrido",
                            f"{resultado['score_hibrido']:.1f}%"
                        )


                    st.write(
                        f"**Planilha:** "
                        f"{resultado['planilha']}"
                    )


                    if resultado["termos"]:

                        st.write(
                            "**Termos técnicos encontrados:** "
                            +
                            ", ".join(
                                resultado["termos"]
                            )
                        )


                    with st.expander(
                        "👁️ Visualizar conteúdo da APR"
                    ):

                        texto = (
                            resultado["texto"]
                        )


                        if len(texto) > 5000:

                            texto = (
                                texto[:5000]
                                +
                                "\n\n"
                                "[Conteúdo reduzido "
                                "para visualização]"
                            )


                        st.text(
                            texto
                        )


                    st.divider()


# ============================================================
# RODAPÉ
# ============================================================

st.caption(
    "POC Gerador de APR com IA — V5.1 | "
    "Ranking Técnico Inteligente"
)
