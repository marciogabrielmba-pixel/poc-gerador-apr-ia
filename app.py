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
    "V5 — Busca semântica híbrida: combina similaridade por significado "
    "com relevância técnica por palavras-chave."
)


# ============================================================
# METAS DO POC
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
    "a", "o", "as", "os", "um", "uma", "uns", "umas",
    "de", "da", "do", "das", "dos",
    "em", "no", "na", "nos", "nas",
    "por", "para", "com", "sem",
    "e", "ou",
    "que", "se",
    "ao", "aos", "à", "às",
    "é", "ser",
    "como",
    "mais", "menos",
    "sobre",
    "entre",
    "durante",
    "após",
    "antes",
    "até",
    "pelo", "pela", "pelos", "pelas"
}


# ============================================================
# TERMOS MUITO GENÉRICOS
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
    "equipe"
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
# PESO DOS TERMOS
# ============================================================

def peso_termo(termo):

    if termo in TERMOS_GENERICOS:
        return 0.25

    return 1.0


# ============================================================
# SCORE LEXICAL
# ============================================================

def calcular_score_apr(consulta, texto):

    tokens_consulta = obter_tokens(consulta)

    if not tokens_consulta:
        return 0.0, []


    texto_normalizado = normalizar(texto)

    termos_encontrados = []

    pontos = 0
    pontos_maximos = 0


    for termo in tokens_consulta:

        peso = peso_termo(termo)

        pontos_maximos += peso

        padrao = r"\b" + re.escape(termo) + r"\b"

        ocorrencias = len(
            re.findall(
                padrao,
                texto_normalizado
            )
        )

        if ocorrencias > 0:

            termos_encontrados.append(termo)

            # Primeira ocorrência tem peso maior
            pontos += peso

            # Pequeno bônus para repetição
            if ocorrencias >= 3:
                pontos += peso * 0.15


    if pontos_maximos == 0:
        return 0.0, termos_encontrados


    score = (pontos / pontos_maximos) * 100

    score = min(score, 100)

    return score, termos_encontrados


# ============================================================
# LEITURA DOS EXCELS
# ============================================================

def ler_excel(arquivo_bytes, nome_arquivo):

    resultados = []

    try:

        workbook = load_workbook(
            io.BytesIO(arquivo_bytes),
            read_only=True,
            data_only=True
        )

        for nome_planilha in workbook.sheetnames:

            sheet = workbook[nome_planilha]

            linhas = []

            for row in sheet.iter_rows(values_only=True):

                valores = []

                for valor in row:

                    if valor is not None:

                        texto = str(valor).strip()

                        if texto:
                            valores.append(texto)

                if valores:

                    linhas.append(" | ".join(valores))


            texto_planilha = "\n".join(linhas).strip()


            if texto_planilha:

                resultados.append({
                    "arquivo": nome_arquivo,
                    "planilha": nome_planilha,
                    "texto": texto_planilha
                })


        workbook.close()

    except Exception as e:

        st.warning(
            f"Erro ao ler {nome_arquivo}: {e}"
        )


    return resultados


# ============================================================
# CRIAÇÃO DOS CHUNKS SEMÂNTICOS
# ============================================================

def criar_chunks(texto, tamanho=1800, sobreposicao=250):

    texto = str(texto)

    if len(texto) <= tamanho:
        return [texto]

    chunks = []

    inicio = 0

    while inicio < len(texto):

        fim = inicio + tamanho

        chunk = texto[inicio:fim]

        if chunk.strip():
            chunks.append(chunk.strip())

        inicio = fim - sobreposicao

        if inicio < 0:
            inicio = 0

        if inicio >= len(texto):
            break

    return chunks


# ============================================================
# MODELO SEMÂNTICO
# ============================================================

@st.cache_resource
def carregar_modelo():

    modelo = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    return modelo


# ============================================================
# PREPARAR ÍNDICE SEMÂNTICO
# ============================================================

def preparar_indice_semantico(base_aprs, modelo):

    chunks = []

    metadados = []

    for item in base_aprs:

        partes = criar_chunks(
            item["texto"],
            tamanho=1800,
            sobreposicao=250
        )

        for numero, parte in enumerate(partes):

            # Incluímos nome da APR e planilha no texto semântico
            texto_embedding = (
                f"Arquivo: {item['arquivo']}\n"
                f"Planilha: {item['planilha']}\n"
                f"{parte}"
            )

            chunks.append(texto_embedding)

            metadados.append({
                "arquivo": item["arquivo"],
                "planilha": item["planilha"],
                "texto": item["texto"],
                "chunk": numero
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

    return embeddings, metadados


# ============================================================
# BUSCA SEMÂNTICA
# ============================================================

def buscar_semantico(
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


    # Similaridade de cosseno
    similaridades = np.dot(
        embeddings,
        query_embedding
    )


    # Melhor similaridade encontrada por arquivo
    melhores_semanticos = {}


    for indice, similaridade in enumerate(similaridades):

        arquivo = metadados[indice]["arquivo"]

        if (
            arquivo not in melhores_semanticos
            or similaridade > melhores_semanticos[arquivo]["similaridade"]
        ):

            melhores_semanticos[arquivo] = {
                "similaridade": float(similaridade),
                "planilha": metadados[indice]["planilha"],
                "chunk": metadados[indice]["chunk"]
            }


    resultados = []


    for item in base_aprs:

        arquivo = item["arquivo"]

        if arquivo not in melhores_semanticos:
            continue


        similaridade = melhores_semanticos[arquivo]["similaridade"]


        # Score semântico convertido para escala 0-100
        #
        # Não representa probabilidade.
        # É apenas um índice de similaridade.
        score_semantico = max(
            0,
            min(
                100,
                ((similaridade + 1) / 2) * 100
            )
        )


        score_lexical, termos = calcular_score_apr(
            consulta,
            item["texto"]
        )


        # ====================================================
        # SCORE HÍBRIDO
        # 70% semântico
        # 30% lexical
        # ====================================================

        score_hibrido = (
            score_semantico * 0.70
            +
            score_lexical * 0.30
        )


        resultados.append({

            "arquivo": arquivo,

            "planilha": item["planilha"],

            "score_semantico": score_semantico,

            "score_lexical": score_lexical,

            "score_hibrido": score_hibrido,

            "similaridade_bruta": similaridade,

            "termos": termos,

            "texto": item["texto"]

        })


    resultados.sort(
        key=lambda x: x["score_hibrido"],
        reverse=True
    )


    # Dedupe final por arquivo
    finais = []

    arquivos_vistos = set()

    for resultado in resultados:

        if resultado["arquivo"] in arquivos_vistos:
            continue

        arquivos_vistos.add(
            resultado["arquivo"]
        )

        finais.append(resultado)

        if len(finais) >= quantidade:
            break


    return finais


# ============================================================
# SESSÃO
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

st.header("📦 1. Carregar base FONTE")

uploaded_file = st.file_uploader(
    "Selecione o FONTE.zip",
    type=["zip"],
    help="Envie o arquivo ZIP contendo as APRs em Excel."
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
            "Lendo todos os arquivos Excel da base..."
        ):

            base_aprs = []


            with zipfile.ZipFile(
                uploaded_file,
                "r"
            ) as zip_ref:

                arquivos_excel = [
                    nome
                    for nome in zip_ref.namelist()
                    if (
                        nome.lower().endswith(".xlsx")
                        or nome.lower().endswith(".xlsm")
                    )
                    and not nome.startswith("__MACOSX")
                ]


                progresso = st.progress(0)


                total_arquivos = len(
                    arquivos_excel
                )


                for contador, nome in enumerate(
                    arquivos_excel,
                    start=1
                ):

                    try:

                        arquivo_bytes = zip_ref.read(
                            nome
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
                            f"Erro em {nome}: {e}"
                        )


                    progresso.progress(
                        contador / total_arquivos
                    )


        st.session_state.base_aprs = base_aprs

        st.session_state.arquivo_processado = nome_zip

        st.session_state.indice_embeddings = None

        st.session_state.metadados_embeddings = []

        st.session_state.indice_pronto = False


        tempo = time.time() - inicio


        arquivos_unicos = len(
            set(
                item["arquivo"]
                for item in base_aprs
            )
        )


        st.success(
            f"Base indexada: {arquivos_unicos} arquivos "
            f"e {len(base_aprs)} planilhas "
            f"em {tempo:.1f} segundos."
        )


# ============================================================
# STATUS DA BASE
# ============================================================

if st.session_state.base_aprs:

    st.divider()

    st.header("🧠 2. Preparar busca semântica")

    total_aprs = len(
        set(
            item["arquivo"]
            for item in st.session_state.base_aprs
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


    st.warning(
        "A primeira preparação pode demorar alguns minutos, "
        "pois o modelo semântico será carregado e as APRs "
        "serão transformadas em vetores matemáticos."
    )


    if st.button(
        "🚀 PREPARAR BUSCA SEMÂNTICA",
        type="primary"
    ):

        inicio_semantico = time.time()


        try:

            with st.spinner(
                "Carregando modelo semântico multilíngue..."
            ):

                modelo = carregar_modelo()


            st.info(
                "Modelo carregado. Criando índice semântico..."
            )


            with st.spinner(
                "Transformando a base de APRs em vetores..."
            ):

                embeddings, metadados = (
                    preparar_indice_semantico(
                        st.session_state.base_aprs,
                        modelo
                    )
                )


            st.session_state.indice_embeddings = embeddings

            st.session_state.metadados_embeddings = metadados

            st.session_state.indice_pronto = True


            tempo_semantico = (
                time.time()
                -
                inicio_semantico
            )


            st.success(
                f"Busca semântica preparada em "
                f"{tempo_semantico:.1f} segundos."
            )


        except Exception as e:

            st.error(
                "Não foi possível preparar o índice semântico."
            )

            st.exception(e)


# ============================================================
# 3. CONSULTA
# ============================================================

if st.session_state.indice_pronto:

    st.divider()

    st.header("🔎 3. Consultar base de APRs")


    consulta = st.text_input(
        "Descreva a atividade que você deseja encontrar:",
        placeholder=(
            "Ex.: Montagem e fixação de eletrocalhas "
            "em altura"
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
                "Digite uma atividade para realizar a busca."
            )

        else:

            inicio_busca = time.time()


            with st.spinner(
                "Analisando similaridade semântica e relevância técnica..."
            ):

                modelo = carregar_modelo()


                resultados = buscar_semantico(
                    consulta,
                    modelo,
                    st.session_state.indice_embeddings,
                    st.session_state.metadados_embeddings,
                    st.session_state.base_aprs,
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
                    f"📋 {len(resultados)} APRs mais relacionadas"
                )


                st.caption(
                    "O índice de relevância é uma medida técnica "
                    "de similaridade para priorizar as APRs. "
                    "Não representa probabilidade de equivalência."
                )


                for posicao, resultado in enumerate(
                    resultados,
                    start=1
                ):

                    st.markdown(
                        f"### {posicao}. {resultado['arquivo']}"
                    )


                    col1, col2, col3 = st.columns(3)


                    with col1:

                        st.metric(
                            "🧠 Similaridade semântica",
                            f"{resultado['score_semantico']:.1f}%"
                        )


                    with col2:

                        st.metric(
                            "🔎 Relevância lexical",
                            f"{resultado['score_lexical']:.1f}%"
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

                        texto = resultado["texto"]

                        if len(texto) > 5000:
                            texto = texto[:5000] + (
                                "\n\n[Conteúdo reduzido para visualização]"
                            )

                        st.text(texto)


                    st.divider()


# ============================================================
# RODAPÉ
# ============================================================

st.caption(
    "POC Gerador de APR com IA — V5 | "
    "Busca semântica híbrida"
)
