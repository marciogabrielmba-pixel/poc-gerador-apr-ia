
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

st.subheader(
    "POC — Consulta contextual à base de APRs"
)

st.info(
    "V6 — Recuperação semântica + seleção da APR-base "
    "+ extração estruturada do conhecimento técnico."
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
    "e", "ou", "que", "se",
    "ao", "aos", "à", "às",
    "é", "ser", "como",
    "mais", "menos",
    "sobre", "entre",
    "durante", "após",
    "antes", "até",
    "pelo", "pela",
    "pelos", "pelas"
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
    "concretar"
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
    "vala",
    "externo",
    "interno"
}


# ============================================================
# TERMOS GENÉRICOS
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
    "obra"
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
        texto = texto.replace(
            origem,
            destino
        )

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
# VARIANTES
# ============================================================

def variantes_termo(termo):

    variantes = {termo}

    if termo.endswith("s") and len(termo) > 4:
        variantes.add(
            termo[:-1]
        )

    if termo.endswith("coes") and len(termo) > 6:
        variantes.add(
            termo[:-4] + "cao"
        )

    return variantes


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
        "contextos": contextos
    }


# ============================================================
# TERMO ENCONTRADO
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
# SCORE GRUPO
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
# SCORE TÉCNICO
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

    score = (
        score_objeto * 0.60
        +
        score_atividade * 0.25
        +
        score_contexto * 0.15
    )

    return (
        min(score, 100),
        objetos_encontrados,
        atividades_encontradas,
        contextos_encontrados,
        min(score_objeto, 100)
    )


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

    score_objeto, _ = calcular_score_grupo(
        grupos["objetos"],
        titulo
    )

    score_atividade, _ = calcular_score_grupo(
        grupos["atividades"],
        titulo
    )

    score_contexto, _ = calcular_score_grupo(
        grupos["contextos"],
        titulo
    )

    return min(
        (
            score_objeto * 0.60
            +
            score_atividade * 0.25
            +
            score_contexto * 0.15
        ),
        100
    )


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
# LEITURA EXCEL
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
# PREPARAR ÍNDICE
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

    query_embedding = modelo.encode(
        [consulta],
        normalize_embeddings=True,
        show_progress_bar=False
    )[0]

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    )

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

        (
            score_tecnico,
            objetos,
            atividades,
            contextos,
            score_objeto
        ) = calcular_score_tecnico(
            consulta,
            item["texto"]
        )

        score_titulo = (
            calcular_score_titulo(
                consulta,
                arquivo,
                item["planilha"]
            )
        )

        indice = (
            score_objeto * 0.30
            +
            score_titulo * 0.25
            +
            score_tecnico * 0.20
            +
            score_semantico * 0.25
        )

        if (
            objetos
            and atividades
            and contextos
        ):
            indice += 5

        elif objetos and atividades:
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
                objetos,

            "atividades":
                atividades,

            "contextos":
                contextos,

            "texto":
                item["texto"]

        })

    resultados.sort(
        key=lambda x:
            x["score_hibrido"],
        reverse=True
    )

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
# EXTRAÇÃO ESTRUTURADA
# ============================================================

CATEGORIAS_EXTRACAO = {

    "Etapas da atividade": [
        "etapa",
        "passo",
        "sequencia",
        "procedimento",
        "execucao",
        "atividade"
    ],

    "Perigos": [
        "perigo",
        "fonte de perigo",
        "agente"
    ],

    "Riscos": [
        "risco",
        "riscos"
    ],

    "Consequências": [
        "consequencia",
        "consequências",
        "lesao",
        "acidente",
        "danos"
    ],

    "Medidas de controle": [
        "medida de controle",
        "medidas de controle",
        "controle",
        "preventiva",
        "prevencao",
        "prevenção"
    ],

    "EPC": [
        "epc",
        "proteção coletiva",
        "protecao coletiva"
    ],

    "EPI": [
        "epi",
        "equipamento de proteção individual",
        "equipamento de protecao individual"
    ],

    "Requisitos legais": [
        "nr-",
        "nr ",
        "norma",
        "legislação",
        "legislacao",
        "requisito legal"
    ]
}


def classificar_linha(
    linha
):

    linha_normalizada = normalizar(
        linha
    )

    categorias = []

    for categoria, termos in (
        CATEGORIAS_EXTRACAO.items()
    ):

        for termo in termos:

            termo_normalizado = normalizar(
                termo
            )

            if termo_normalizado in linha_normalizada:

                categorias.append(
                    categoria
                )

                break

    return categorias


def extrair_conhecimento(
    texto
):

    linhas = [
        linha.strip()
        for linha
        in str(texto).splitlines()
        if linha.strip()
    ]

    resultado = {
        categoria: []
        for categoria
        in CATEGORIAS_EXTRACAO
    }

    linhas_sem_categoria = []

    for linha in linhas:

        categorias = classificar_linha(
            linha
        )

        if categorias:

            for categoria in categorias:

                if linha not in (
                    resultado[categoria]
                ):

                    resultado[
                        categoria
                    ].append(
                        linha
                    )

        else:

            linhas_sem_categoria.append(
                linha
            )

    resultado[
        "Outras informações"
    ] = linhas_sem_categoria

    return resultado


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

if "resultados_busca" not in st.session_state:
    st.session_state.resultados_busca = []

if "apr_selecionada" not in st.session_state:
    st.session_state.apr_selecionada = None

if "extracao" not in st.session_state:
    st.session_state.extracao = None

if "apr_gerada" not in st.session_state:
    st.session_state.apr_gerada = None


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

        st.session_state.resultados_busca = []

        st.session_state.apr_selecionada = None

        st.session_state.extracao = None
        st.session_state.apr_gerada = None

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

        st.session_state.resultados_busca = []

        st.session_state.apr_selecionada = None

        st.session_state.extracao = None
        st.session_state.apr_gerada = None

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

    if st.button(
        "🚀 PREPARAR BUSCA SEMÂNTICA",
        type="primary"
    ):

        inicio = time.time()

        try:

            with st.spinner(
                "Carregando modelo semântico..."
            ):

                modelo = carregar_modelo()

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

            st.session_state.indice_pronto = True

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
        "🔎 3. Encontrar APRs relacionadas"
    )

    consulta = st.text_input(
        "Descreva a atividade:",
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
        "🔍 BUSCAR APRs",
        type="primary"
    ):

        if not consulta.strip():

            st.warning(
                "Digite uma atividade."
            )

        else:

            inicio = time.time()

            with st.spinner(
                "Analisando objeto, atividade, "
                "contexto e semântica..."
            ):

                modelo = carregar_modelo()

                resultados = buscar_aprs(
                    consulta,
                    modelo,
                    st.session_state.indice_embeddings,
                    st.session_state.metadados_embeddings,
                    st.session_state.base_aprs,
                    quantidade
                )

            st.session_state.resultados_busca = (
                resultados
            )

            st.session_state.apr_selecionada = None

            st.session_state.extracao = None
            st.session_state.apr_gerada = None

            tempo = (
                time.time()
                -
                inicio
            )

            st.success(
                f"Busca concluída em "
                f"{tempo:.2f} segundos."
            )


# ============================================================
# RESULTADOS
# ============================================================

if st.session_state.resultados_busca:

    st.divider()

    st.header(
        "📋 4. APRs encontradas"
    )

    for posicao, resultado in enumerate(
        st.session_state.resultados_busca,
        start=1
    ):

        st.markdown(
            f"### {posicao}. "
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
                "**Objetos:** "
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
            "👁️ Visualizar conteúdo"
        ):

            texto = resultado["texto"]

            if len(texto) > 5000:

                texto = (
                    texto[:5000]
                    +
                    "\n\n"
                    "[Conteúdo reduzido]"
                )

            st.text(texto)

        st.divider()


    # ========================================================
    # SELEÇÃO DA APR-BASE
    # ========================================================

    st.header(
        "🎯 5. Selecionar APR-base"
    )

    opcoes = []

    for indice, resultado in enumerate(
        st.session_state.resultados_busca
    ):

        opcoes.append(
            f"{indice + 1}. "
            f"{resultado['arquivo']}"
        )

    selecao = st.selectbox(
        "Escolha a APR que será utilizada como base técnica:",
        options=opcoes
    )


    indice_selecionado = opcoes.index(
        selecao
    )

    resultado_selecionado = (
        st.session_state.resultados_busca[
            indice_selecionado
        ]
    )


    st.info(
        f"APR-base selecionada: "
        f"**{resultado_selecionado['arquivo']}**"
    )


    if st.button(
        "🧩 EXTRAIR CONHECIMENTO DA APR",
        type="primary"
    ):

        inicio_extracao = time.time()

        with st.spinner(
            "Extraindo informações estruturadas "
            "da APR selecionada..."
        ):

            extracao = extrair_conhecimento(
                resultado_selecionado["texto"]
            )

        st.session_state.apr_selecionada = (
            resultado_selecionado
        )

        st.session_state.extracao = (
            extracao
        )
        st.session_state.apr_gerada = None

        tempo_extracao = (
            time.time()
            -
            inicio_extracao
        )

        st.success(
            f"Extração concluída em "
            f"{tempo_extracao:.2f} segundos."
        )


# ============================================================
# 6. CONHECIMENTO EXTRAÍDO
# ============================================================

if st.session_state.extracao:

    st.divider()

    st.header(
        "🧠 6. Conhecimento técnico extraído"
    )

    apr = (
        st.session_state.apr_selecionada
    )

    st.success(
        f"Fonte utilizada: "
        f"{apr['arquivo']} — "
        f"{apr['planilha']}"
    )


    # --------------------------------------------------------
    # RESUMO DA EXTRAÇÃO
    # --------------------------------------------------------

    extracao = (
        st.session_state.extracao
    )


    categorias_principais = [
        "Etapas da atividade",
        "Perigos",
        "Riscos",
        "Consequências",
        "Medidas de controle",
        "EPC",
        "EPI",
        "Requisitos legais"
    ]


    for categoria in categorias_principais:

        itens = extracao.get(
            categoria,
            []
        )


        st.subheader(
            categoria
        )


        if not itens:

            st.caption(
                "Nenhuma informação "
                "identificada explicitamente "
                "na estrutura textual analisada."
            )

        else:

            for item in itens:

                st.markdown(
                    f"- {item}"
                )


    # --------------------------------------------------------
    # OUTRAS INFORMAÇÕES
    # --------------------------------------------------------

    outras = extracao.get(
        "Outras informações",
        []
    )


    if outras:

        with st.expander(
            "📄 Outras informações da APR"
        ):

            for item in outras:

                st.markdown(
                    f"- {item}"
                )


    # --------------------------------------------------------
    # AVISO DE RASTREABILIDADE
    # --------------------------------------------------------

    st.warning(
        "⚠️ V6: esta etapa organiza informações "
        "encontradas na APR selecionada. "
        "Ela não cria, valida ou aprova riscos, "
        "medidas de controle ou requisitos legais "
        "que não estejam explicitamente presentes "
        "na fonte."
    )


# ============================================================
# 7. GERAR APR — V7.1
# ============================================================

if st.session_state.extracao and st.session_state.apr_selecionada:

    st.divider()
    st.header("📝 7. Gerar APR")
    st.info(
        "V7.1 — Geração estruturada assistida a partir do conhecimento "
        "extraído da APR-base. O conteúdo abaixo permanece rastreável à fonte."
    )

    extracao = st.session_state.extracao
    apr_base = st.session_state.apr_selecionada

    # --------------------------------------------------------
    # FUNÇÕES DA V7.1
    # --------------------------------------------------------

    def limpar_item(item):
        """Limpa marcadores comuns sem alterar o conteúdo técnico."""
        item = str(item).strip()
        item = re.sub(r"^[-•▪◦]+\s*", "", item)
        return item.strip()


    def tokens_relevantes(texto):
        return set(obter_tokens(texto))


    def similaridade_textual(a, b):
        """Similaridade lexical simples usada apenas para ordenar itens da fonte."""
        ta = tokens_relevantes(a)
        tb = tokens_relevantes(b)
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / max(1, len(ta | tb))


    def selecionar_itens_relevantes(tarefa, itens, limite=3):
        itens = [limpar_item(x) for x in itens if str(x).strip()]
        if not itens:
            return []
        ordenados = sorted(
            enumerate(itens),
            key=lambda par: (similaridade_textual(tarefa, par[1]), -par[0]),
            reverse=True
        )
        selecionados = [item for _, item in ordenados[:limite]]
        return selecionados


    def classificar_barreira(texto, origem=""):
        """Classificação heurística transparente das medidas já presentes na fonte."""
        t = normalizar(texto)
        o = normalizar(origem)

        apoio = (
            "emergencia", "resgate", "primeiros socorros", "limpeza",
            "organizacao", "organizacao do local", "apoio", "comunicacao",
            "observador", "vigia", "sinalizacao de apoio"
        )
        protecao = (
            "epi", "capacete", "luva", "oculos", "protetor", "cinturao",
            "talabarte", "trava quedas", "linha de vida", "guarda corpo",
            "rodape", "plataforma", "andaime", "escada", "isolamento fisico",
            "cone", "fita zebrada", "barreira fisica", "epc"
        )
        controle = (
            "inspecao", "inspecionar", "procedimento", "treinamento", "treinado",
            "autorizacao", "permissao", "planejamento", "analise", "apr",
            "isolamento", "sinalizacao", "bloqueio", "desenergizacao", "desligamento",
            "checklist", "supervisao", "orientacao", "comunicacao de risco",
            "seguir", "verificar", "conferir", "manter distancia"
        )

        if any(term in t for term in apoio):
            return "Barreira de apoio"
        if any(term in t for term in protecao) or "EPC" in origem.upper() or "EPI" in origem.upper():
            return "Barreira de proteção"
        if any(term in t for term in controle):
            return "Barreira de controle"
        return "Barreira de controle"


    def construir_apr_v71(extracao):
        etapas = [limpar_item(x) for x in extracao.get("Etapas da atividade", []) if str(x).strip()]
        riscos = [limpar_item(x) for x in extracao.get("Riscos", []) if str(x).strip()]
        if not riscos:
            riscos = [limpar_item(x) for x in extracao.get("Perigos", []) if str(x).strip()]

        controles = [
            (limpar_item(x), "Medidas de controle")
            for x in extracao.get("Medidas de controle", []) if str(x).strip()
        ]
        controles += [
            (limpar_item(x), "EPC")
            for x in extracao.get("EPC", []) if str(x).strip()
        ]
        controles += [
            (limpar_item(x), "EPI")
            for x in extracao.get("EPI", []) if str(x).strip()
        ]

        if not etapas:
            etapas = [
                "Etapa não identificada explicitamente na APR-base; revisar antes da emissão."
            ]

        linhas = []
        for i, etapa in enumerate(etapas[:15]):
            riscos_rel = selecionar_itens_relevantes(etapa, riscos, limite=2)
            if not riscos_rel:
                riscos_rel = [
                    "Risco não identificado explicitamente para esta etapa na APR-base."
                ]

            controles_rel = sorted(
                controles,
                key=lambda par: similaridade_textual(etapa, par[0]),
                reverse=True
            )[:8]

            grupos = {
                "Barreira de controle": [],
                "Barreira de proteção": [],
                "Barreira de apoio": []
            }
            for controle, origem in controles_rel:
                grupo = classificar_barreira(controle, origem)
                if controle not in grupos[grupo]:
                    grupos[grupo].append(controle)

            linhas.append({
                "tarefa": etapa,
                "risco": " | ".join(riscos_rel),
                "nivel": "Não informado",
                "controle": " | ".join(grupos["Barreira de controle"]),
                "protecao": " | ".join(grupos["Barreira de proteção"]),
                "apoio": " | ".join(grupos["Barreira de apoio"])
            })

        return linhas


    # --------------------------------------------------------
    # GERAR PRÉVIA
    # --------------------------------------------------------

    if st.button("🚀 GERAR PRÉVIA DA APR", type="primary"):
        inicio_geracao = time.time()
        with st.spinner("Organizando tarefas, riscos e barreiras a partir da APR-base..."):
            st.session_state.apr_gerada = construir_apr_v71(extracao)
        tempo_geracao = time.time() - inicio_geracao
        st.success(f"Prévia da APR gerada em {tempo_geracao:.2f} segundos.")


    if "apr_gerada" in st.session_state and st.session_state.apr_gerada:

        st.subheader("📋 Prévia da APR gerada")
        st.caption(
            f"Fonte: {apr_base['arquivo']} — {apr_base['planilha']} | "
            "A classificação das barreiras é uma sugestão da POC baseada no texto da fonte."
        )

        nivel_opcoes = ["Não informado", "A", "M", "B"]

        for idx, linha in enumerate(st.session_state.apr_gerada):
            with st.container(border=True):
                st.markdown(f"**Tarefa {idx + 1} — {linha['tarefa']}**")
                st.markdown(f"**Risco:** {linha['risco']}")

                nivel = st.selectbox(
                    "Nível (A/M/B)",
                    nivel_opcoes,
                    index=nivel_opcoes.index(linha.get("nivel", "Não informado")),
                    key=f"nivel_apr_v71_{idx}"
                )
                st.session_state.apr_gerada[idx]["nivel"] = nivel

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**Barreira de controle**")
                    st.write(linha["controle"] or "Não identificada explicitamente na fonte.")
                with col2:
                    st.markdown("**Barreira de proteção**")
                    st.write(linha["protecao"] or "Não identificada explicitamente na fonte.")
                with col3:
                    st.markdown("**Barreira de apoio**")
                    st.write(linha["apoio"] or "Não identificada explicitamente na fonte.")

        # ----------------------------------------------------
        # EXPORTAÇÃO EXCEL
        # ----------------------------------------------------

        st.divider()
        st.subheader("📥 Exportar APR")

        from openpyxl import Workbook
        from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
        from openpyxl.utils import get_column_letter

        def criar_excel_apr_v71():
            wb = Workbook()
            ws = wb.active
            ws.title = "APR Gerada"

            headers = [
                "Descrição das tarefas (passo a passo)",
                "Riscos associados às tarefas",
                "Nível (A/M/B)",
                "Barreira de controle",
                "Barreira de proteção",
                "Barreira de apoio"
            ]

            ws.append(headers)

            for linha in st.session_state.apr_gerada:
                ws.append([
                    linha["tarefa"],
                    linha["risco"],
                    linha["nivel"],
                    linha["controle"],
                    linha["protecao"],
                    linha["apoio"]
                ])

            # Cabeçalho e layout
            header_fill = PatternFill(fill_type="solid", fgColor="1F4E78")
            header_font = Font(color="FFFFFF", bold=True)
            thin = Side(style="thin", color="B7B7B7")

            for cell in ws[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                cell.border = Border(left=thin, right=thin, top=thin, bottom=thin)

            for row in ws.iter_rows(min_row=2):
                for cell in row:
                    cell.alignment = Alignment(vertical="top", wrap_text=True)
                    cell.border = Border(left=thin, right=thin, top=thin, bottom=thin)

            larguras = [38, 42, 14, 48, 48, 42]
            for i, largura in enumerate(larguras, start=1):
                ws.column_dimensions[get_column_letter(i)].width = largura

            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions

            # Aba de rastreabilidade
            fonte = wb.create_sheet("Rastreabilidade")
            fonte.append(["Campo", "Informação"])
            fonte.append(["APR-base", apr_base["arquivo"]])
            fonte.append(["Planilha", apr_base["planilha"]])
            fonte.append(["Índice V5.2", round(apr_base.get("score_hibrido", 0), 1)])
            fonte.append(["Observação", "Conteúdo estruturado a partir das informações extraídas da APR-base; revisar e aprovar antes da emissão."])

            for cell in fonte[1]:
                cell.fill = header_fill
                cell.font = header_font
            fonte.column_dimensions["A"].width = 22
            fonte.column_dimensions["B"].width = 90
            for row in fonte.iter_rows():
                for cell in row:
                    cell.alignment = Alignment(vertical="top", wrap_text=True)

            buffer = io.BytesIO()
            wb.save(buffer)
            buffer.seek(0)
            return buffer.getvalue()

        arquivo_excel = criar_excel_apr_v71()

        st.download_button(
            label="⬇️ BAIXAR APR GERADA EM EXCEL",
            data=arquivo_excel,
            file_name="APR_Gerada_V7_1.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary"
        )

        st.warning(
            "⚠️ V7.1: a prévia é uma organização assistida do conteúdo encontrado na APR-base. "
            "O sistema não deve ser tratado como aprovação técnica. O responsável pela APR "
            "deve revisar tarefas, riscos, nível e barreiras antes da emissão."
        )


# ============================================================
# RODAPÉ
# ============================================================

st.caption(
    "POC Gerador de APR com IA — V7.1 | "
    "Busca + Seleção + Extração + Geração Estruturada"
)
