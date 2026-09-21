import streamlit as st
import zipfile
import io
import re
import time
import numpy as np
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
    "V5.2 — Ranking inteligente por objeto técnico, "
    "atividade, contexto e similaridade semântica."
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
    "ao", "aos", "à", "às",
    "é", "ser",
    "como", "mais", "menos",
    "sobre", "entre",
    "durante", "após",
    "antes", "até",
    "pelo", "pela",
    "pelos", "pelas",
    "um", "uma"
}


# ============================================================
# TERMOS DE ATIVIDADE
# ============================================================

TERMOS_ATIVIDADE = {
    "instalacao",
    "instalar",
    "montagem",
    "montar",
    "fixacao",
    "fixar",
    "execucao",
    "executar",
    "manutencao",
    "manter",
    "passagem",
    "passar",
    "lancamento",
    "lancar",
    "montagem",
    "desmontagem",
    "desmontar",
    "fabricacao",
    "fabricar",
    "soldagem",
    "soldar",
    "corte",
    "cortar",
    "perfuracao",
    "perfurar",
    "transporte",
    "movimentacao",
    "escavacao",
    "escavar",
    "concretagem",
    "concretar",
    "montagem"
}


# ============================================================
# TERMOS DE CONTEXTO
# ============================================================

TERMOS_CONTEXTO = {
    "altura",
    "alto",
    "elevado",
    "elevacao",
    "telhado",
    "cobertura",
    "forro",
    "andaime",
    "escada",
    "pta",
    "plataforma",
    "espaco",
    "confinado",
    "eletrico",
    "eletrica",
    "eletricas",
    "energia",
    "tensao",
    "baixa",
    "media",
    "alta",
    "solo",
    "subterraneo",
    "escavacao",
    "vala",
    "externo",
    "interno"
}


# ============================================================
# TERMOS MUITO GENÉRICOS
# ============================================================

TERMOS_GENERICOS = {
    "servico",
    "atividade",
    "trabalho",
    "sistema",
    "processo",
    "realizacao",
    "operacao",
    "estrutura",
    "equipamento",
    "procedimento",
    "local",
    "material",
    "equipe",
    "obra",
    "execucao"
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

    texto = re.sub(
        r"[^a-z0-9\s]",
        " ",
        texto
    )

    texto = re.sub(
        r"\s+",
        " ",
        texto
    )

    return texto.strip()


# ============================================================
# VARIANTES DE TERMOS
# ============================================================

def variantes_termo(termo):

    variantes = {termo}

    if termo.endswith("s") and len(termo) > 4:
        variantes.add(
            termo[:-1]
        )

    # instalações -> instalacao
    if termo.endswith("coes") and len(termo) > 6:
        variantes.add(
            termo[:-4] + "cao"
        )

    return variantes


# ============================================================
# TOKENS
# ============================================================

def obter_tokens(texto):

    tokens = normalizar(
        texto
    ).split()

    return [
        token
        for token in tokens
        if token not in STOPWORDS
        and len(token) > 2
    ]


# ============================================================
# CLASSIFICAÇÃO DA CONSULTA
# ============================================================

def classificar_consulta(consulta):

    tokens = obter_tokens(
        consulta
    )

    objetos = []
    atividades = []
    contextos = []
    outros = []

    for token in tokens:

        if token in TERMOS_ATIVIDADE:
            atividades.append(token)

        elif token in TERMOS_CONTEXTO:
            contextos.append(token)

        elif token in TERMOS_GENERICOS:
            continue

        else:
            objetos.append(token)

    return {
        "objetos": objetos,
        "atividades": atividades,
        "contextos": contextos,
        "outros": outros
    }


# ============================================================
# VERIFICAÇÃO DE TERMO
# ============================================================

def termo_encontrado(
    termo,
    texto
):

    texto_normalizado = normalizar(
        texto
    )

    for variante in variantes_termo(
        termo
    ):

        if re.search(
            r"\b"
            + re.escape(variante)
            + r"\b",
            texto_normalizado
        ):
            return True

    return False


# ============================================================
# SCORE DE GRUPO
# ============================================================

def calcular_score_grupo(
    termos,
    texto
):

    if not termos:
        return 0.0, []

    encontrados = []

    for termo in termos:

        if termo_encontrado(
            termo,
            texto
        ):

            encontrados.append(
                termo
            )

    score = (
        len(encontrados)
        /
        len(termos)
    ) * 100

    return score, encontrados


# ============================================================
# SCORE DO TÍTULO
# ============================================================

def calcular_score_titulo(
    consulta,
    arquivo,
    planilha
):

    titulo = (
        f"{arquivo} {planilha}"
    )

    grupos = classificar_consulta(
        consulta
    )

    objetos = grupos["objetos"]
    atividades = grupos["atividades"]
    contextos = grupos["contextos"]


    score_objeto, _ = calcular_score_grupo(
        objetos,
        titulo
    )

    score_atividade, _ = calcular_score_grupo(
        atividades,
        titulo
    )

    score_contexto, _ = calcular_score_grupo(
        contextos,
        titulo
    )


    # O título recebe peso muito alto
    score = (
        score_objeto * 0.60
        +
        score_atividade * 0.25
        +
        score_contexto * 0.15
    )


    return min(
        score,
        100
    )


# ============================================================
# SCORE TÉCNICO DO CONTEÚDO
# ============================================================

def calcular_score_tecnico(
    consulta,
    texto
):

    grupos = classificar_consulta(
        consulta
    )

    objetos = grupos["objetos"]
    atividades = grupos["atividades"]
    contextos = grupos["contextos"]


    score_objeto, objetos_encontrados = (
        calcular_score_grupo(
            objetos,
            texto
        )
    )

    score_atividade, atividades_encontradas = (
        calcular_score_grupo(
            atividades,
            texto
        )
    )

    score_contexto, contextos_encontrados = (
        calcular_score_grupo(
            contextos,
            texto
        )
    )


    # OBJETO é o elemento mais importante.
    score = (
        score_objeto * 0.60
        +
        score_atividade * 0.25
        +
        score_contexto * 0.15
    )


    termos_encontrados = (
        objetos_encontrados
        +
        atividades_encontradas
        +
        contextos_encontrados
    )


    return (
        min(score, 100),
        objetos_encontrados,
        atividades_encontradas,
        contextos_encontrados,
        min(score_objeto, 100)
    )


# ============================================================
# PENALIZAÇÃO POR AUSÊNCIA DO OBJETO
# ============================================================

def calcular_penalizacao_objeto(
    objetos,
    arquivo,
    planilha,
    texto
):

    if not objetos:
        return 0

    titulo = (
        f"{arquivo} {planilha}"
    )

    encontrados_titulo = [
        termo
        for termo in objetos
        if termo_encontrado(
            termo,
            titulo
        )
    ]


    encontrados_texto = [
        termo
        for termo in objetos
        if termo_encontrado(
            termo,
            texto
        )
    ]


    # Se nenhum objeto técnico aparece,
    # aplicamos penalização forte.
    if not encontrados_texto:

        return 30


    # Se aparece no conteúdo mas não no título,
    # penalização moderada.
    if not encontrados_titulo:

        return 8


    return 0


# ============================================================
# MODELO SEMÂNTICO
# ============================================================

@st.cache_resource
def carregar_modelo():

    return SentenceTransformer(
        "sentence-transformers/"
        "paraphrase-multilingual-MiniLM-L12-v2"
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


    return (
        np.asarray(
            embeddings,
            dtype=np.float32
        ),
        metadados
    )


# ============================================================
# BUSCA V5.2
# ============================================================

def buscar_aprs(
    consulta,
    modelo,
    embeddings,
    metadados,
    base_aprs,
    quantidade=10
):

    # --------------------------------------------------------
    # EMBEDDING DA CONSULTA
    # --------------------------------------------------------

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
                    ]["planilha"]

            }


    # --------------------------------------------------------
    # CLASSIFICAÇÃO DA CONSULTA
    # --------------------------------------------------------

    grupos = classificar_consulta(
        consulta
    )

    objetos = grupos["objetos"]
    atividades = grupos["atividades"]
    contextos = grupos["contextos"]


    resultados = []


    # --------------------------------------------------------
    # AVALIAR CADA APR
    # --------------------------------------------------------

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


        # -----------------------------------------------
        # SEMÂNTICA
        # -----------------------------------------------

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


        # -----------------------------------------------
        # TÉCNICO
        # -----------------------------------------------

        (
            score_tecnico,
            objetos_encontrados,
            atividades_encontradas,
            contextos_encontrados,
            score_objeto
        ) = calcular_score_tecnico(
            consulta,
            item["texto"]
        )


        # -----------------------------------------------
        # TÍTULO
        # -----------------------------------------------

        score_titulo = (
            calcular_score_titulo(
                consulta,
                arquivo,
                item["planilha"]
            )
        )


        # -----------------------------------------------
        # PENALIZAÇÃO
        # -----------------------------------------------

        penalizacao = (
            calcular_penalizacao_objeto(
                objetos,
                arquivo,
                item["planilha"],
                item["texto"]
            )
        )


        # -----------------------------------------------
        # ÍNDICE V5.2
        # -----------------------------------------------
        #
        # Objeto técnico:      30%
        # Título da APR:       25%
        # Conteúdo técnico:   20%
        # Semântica:           25%
        #
        # -----------------------------------------------

        indice = (

            score_objeto * 0.30

            +

            score_titulo * 0.25

            +

            score_tecnico * 0.20

            +

            score_semantico * 0.25

        )


        indice -= penalizacao


        # -----------------------------------------------
        # BÔNUS DE CONVERGÊNCIA
        # -----------------------------------------------

        # Se objeto + atividade + contexto aparecem,
        # temos forte convergência.

        if (
            objetos_encontrados
            and
            atividades_encontradas
            and
            contextos_encontrados
        ):

            indice += 5


        elif (
            objetos_encontrados
            and
            atividades_encontradas
        ):

            indice += 3


        indice = max(
            0,
            min(
                indice,
                100
            )
        )


        resultados.append({

            "arquivo":
                arquivo,

            "planilha":
                item["planilha"],

            "score_semantico":
                score_semantico,

            "score_objeto":
                score_objeto,

            "score_tecnico":
                score_tecnico,

            "score_titulo":
                score_titulo,

            "score_hibrido":
                indice,

            "objetos":
                objetos_encontrados,

            "atividades":
                atividades_encontradas,

            "contextos":
                contextos_encontrados,

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
    # DEDUPLICAR POR ARQUIVO
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
# 2. BUSCA SEMÂNTICA
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
        "V5.2: o ranking considera "
        "objeto técnico, atividade, contexto, "
        "nome da APR e similaridade semântica."
    )


    if st.button(
        "🚀 PREPARAR BUSCA SEMÂNTICA",
        type="primary"
    ):

        inicio = time.time()


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


            tempo = (
                time.time()
                -
                inicio
            )


            st.success(
                f"Busca semântica preparada "
                f"em {tempo:.1f} segundos."
            )


        except Exception as e:

            st.error(
                "Erro ao preparar "
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

            inicio = time.time()


            with st.spinner(
                "Analisando objeto, atividade, "
                "contexto e semântica..."
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

                    quantidade

                )


            tempo = (
                time.time()
                -
                inicio
            )


            st.success(
                f"Busca concluída em "
                f"{tempo:.2f} segundos."
            )


            grupos = classificar_consulta(
                consulta
            )


            st.info(
                "🎯 **Interpretação da consulta** — "
                f"Objetos: {', '.join(grupos['objetos']) or 'nenhum'} | "
                f"Atividades: {', '.join(grupos['atividades']) or 'nenhuma'} | "
                f"Contextos: {', '.join(grupos['contextos']) or 'nenhum'}"
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
                    "O índice V5.2 é um indicador "
                    "de priorização técnica. "
                    "Não representa probabilidade "
                    "de equivalência nem aprovação "
                    "da APR."
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


                    col1, col2, col3, col4 = (
                        st.columns(4)
                    )


                    with col1:

                        st.metric(
                            "🎯 Objeto",
                            f"{resultado['score_objeto']:.1f}%"
                        )


                    with col2:

                        st.metric(
                            "📝 Título",
                            f"{resultado['score_titulo']:.1f}%"
                        )


                    with col3:

                        st.metric(
                            "🧠 Semântica",
                            f"{resultado['score_semantico']:.1f}%"
                        )


                    with col4:

                        st.metric(
                            "⭐ Índice V5.2",
                            f"{resultado['score_hibrido']:.1f}%"
                        )


                    st.write(
                        f"**Planilha:** "
                        f"{resultado['planilha']}"
                    )


                    if resultado["objetos"]:

                        st.write(
                            "**Objetos técnicos:** "
                            +
                            ", ".join(
                                resultado["objetos"]
                            )
                        )


                    if resultado["atividades"]:

                        st.write(
                            "**Atividades:** "
                            +
                            ", ".join(
                                resultado["atividades"]
                            )
                        )


                    if resultado["contextos"]:

                        st.write(
                            "**Contextos:** "
                            +
                            ", ".join(
                                resultado["contextos"]
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
    "POC Gerador de APR com IA — V5.2 | "
    "Objeto + Atividade + Contexto + Semântica"
)
