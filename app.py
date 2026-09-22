import streamlit as st
import zipfile
import io
import re
import time
import numpy as np

from copy import copy
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

st.title("🦺 Gerador de APR com IA")

st.subheader(
    "POC — Busca, estruturação e geração de APR"
)

st.info(
    "V7.1 — preserva a relação Tarefa → Risco → Nível → "
    "Barreiras antes de gerar o Excel."
)


# ============================================================
# FUNÇÕES BÁSICAS
# ============================================================

STOPWORDS = {
    "a", "o", "as", "os", "um", "uma",
    "de", "da", "do", "das", "dos",
    "em", "no", "na", "nos", "nas",
    "por", "para", "com", "sem",
    "e", "ou", "que", "se",
    "ao", "aos", "à", "às",
    "é", "ser", "como",
    "mais", "menos",
    "sobre", "entre",
    "durante", "após", "antes",
    "até", "pelo", "pela",
    "pelos", "pelas"
}


def normalizar(texto):

    texto = str(texto or "").lower()

    substituicoes = {
        "á": "a",
        "à": "a",
        "ã": "a",
        "â": "a",
        "é": "e",
        "ê": "e",
        "í": "i",
        "ó": "o",
        "õ": "o",
        "ô": "o",
        "ú": "u",
        "ç": "c"
    }

    for origem, destino in substituicoes.items():
        texto = texto.replace(
            origem,
            destino
        )

    texto = re.sub(
        r"[^a-z0-9\s|:/().\-]",
        " ",
        texto
    )

    texto = re.sub(
        r"\s+",
        " ",
        texto
    )

    return texto.strip()


def tokens(texto):

    return [
        x
        for x in normalizar(texto)
        .replace("|", " ")
        .split()
        if x not in STOPWORDS
        and len(x) > 2
    ]


def remover_duplicados(lista):

    resultado = []
    vistos = set()

    for item in lista:

        chave = normalizar(item)

        if not chave:
            continue

        if chave in vistos:
            continue

        vistos.add(chave)
        resultado.append(item)

    return resultado


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
    dados,
    nome_arquivo
):

    workbook = load_workbook(
        io.BytesIO(dados),
        read_only=True,
        data_only=True
    )

    resultado = []

    for nome_planilha in workbook.sheetnames:

        ws = workbook[
            nome_planilha
        ]

        linhas = []

        for row in ws.iter_rows(
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
                    valores
                )

        texto_completo = "\n".join(
            " | ".join(linha)
            for linha in linhas
        )

        registros = extrair_registros(
            linhas
        )

        resultado.append({

            "arquivo":
                nome_arquivo,

            "planilha":
                nome_planilha,

            "texto":
                texto_completo,

            "registros":
                registros

        })

    workbook.close()

    return resultado


# ============================================================
# UTILITÁRIOS DE ESTRUTURAÇÃO
# ============================================================

def eh_numero(valor):

    return bool(
        re.fullmatch(
            r"\d+(?:[.,]\d+)?",
            str(valor).strip()
        )
    )


def nivel_explicito(texto):

    texto = normalizar(
        texto
    )

    padroes = [

        r"\bnivel\s*[:\-]?\s*([amb])\b",

        r"\b([amb])\s*[-–]\s*"
        r"(alto|medio|baixo)\b",

        r"\b([amb])\s*\("
    ]

    for padrao in padroes:

        resultado = re.search(
            padrao,
            texto,
            re.IGNORECASE
        )

        if resultado:

            return resultado.group(
                1
            ).upper()

    return ""


def procurar_cabecalho(linha):

    texto = normalizar(
        " | ".join(linha)
    )

    indicadores = [

        (
            "descricao das tarefas"
            in texto
        ),

        (
            "riscos associados"
            in texto
        ),

        (
            "barreira de controle"
            in texto
        ),

        (
            "medidas de controle"
            in texto
        )

    ]

    return sum(indicadores) >= 2


def mapear_colunas(linha):

    mapa = {}

    for indice, valor in enumerate(
        linha
    ):

        texto = normalizar(
            valor
        )

        if (
            "descricao das tarefas"
            in texto
            or
            "sequencia da tarefa"
            in texto
        ):

            mapa["tarefa"] = indice

        elif (
            "riscos associados"
            in texto
            or
            "perigos" in texto
        ):

            mapa["risco"] = indice

        elif (
            "nivel" in texto
            and
            (
                "a/m/b" in texto
                or
                "amb" in texto
            )
        ):

            mapa["nivel"] = indice

        elif (
            "barreira de controle"
            in texto
            or
            "medidas de controle"
            in texto
        ):

            mapa["controle"] = indice

        elif (
            "barreira de protecao"
            in texto
            or
            "epi" in texto
        ):

            mapa["protecao"] = indice

        elif (
            "barreira de apoio"
            in texto
            or
            "procedimento de seguranca"
            in texto
        ):

            mapa["apoio"] = indice

    return mapa


# ============================================================
# EXTRAÇÃO ESTRUTURADA
# ============================================================

def extrair_registros(linhas):

    registros = []

    mapa = None
    ultima_tarefa = ""

    for linha in linhas:

        if not linha:
            continue

        # ----------------------------------------------------
        # Detecta cabeçalho
        # ----------------------------------------------------

        if procurar_cabecalho(
            linha
        ):

            mapa = mapear_colunas(
                linha
            )

            continue

        # ----------------------------------------------------
        # Estrutura tabular
        # ----------------------------------------------------

        if mapa:

            tarefa = ""

            risco = ""

            nivel = ""

            controle = ""

            protecao = ""

            apoio = ""


            if "tarefa" in mapa:

                i = mapa["tarefa"]

                if i < len(linha):

                    tarefa = (
                        linha[i]
                        or ""
                    )


            if "risco" in mapa:

                i = mapa["risco"]

                if i < len(linha):

                    risco = (
                        linha[i]
                        or ""
                    )


            if "nivel" in mapa:

                i = mapa["nivel"]

                if i < len(linha):

                    nivel = (
                        linha[i]
                        or ""
                    )


            if "controle" in mapa:

                i = mapa["controle"]

                if i < len(linha):

                    controle = (
                        linha[i]
                        or ""
                    )


            if "protecao" in mapa:

                i = mapa["protecao"]

                if i < len(linha):

                    protecao = (
                        linha[i]
                        or ""
                    )


            if "apoio" in mapa:

                i = mapa["apoio"]

                if i < len(linha):

                    apoio = (
                        linha[i]
                        or ""
                    )


            if tarefa:

                ultima_tarefa = (
                    tarefa
                )


            if risco:

                registros.append({

                    "tarefa":
                        tarefa
                        or
                        ultima_tarefa,

                    "risco":
                        risco,

                    "nivel":
                        nivel,

                    "controle":
                        controle,

                    "protecao":
                        protecao,

                    "apoio":
                        apoio

                })

            continue


        # ----------------------------------------------------
        # Estrutura legada:
        #
        # Tarefa | Risco | P | S | R | Controle
        # ----------------------------------------------------

        if len(linha) >= 4:

            for indice, valor in enumerate(
                linha
            ):

                if not eh_numero(
                    valor
                ):

                    continue

                if (
                    indice + 2
                    >= len(linha)
                ):

                    continue

                if not eh_numero(
                    linha[indice + 1]
                ):

                    continue

                if not eh_numero(
                    linha[indice + 2]
                ):

                    continue


                risco = ""

                if indice > 0:

                    risco = (
                        linha[
                            indice - 1
                        ]
                    )


                controle = ""

                if (
                    indice + 3
                    < len(linha)
                ):

                    controle = (
                        " | ".join(
                            linha[
                                indice + 3:
                            ]
                        )
                    )


                if risco:

                    registros.append({

                        "tarefa":
                            ultima_tarefa,

                        "risco":
                            risco,

                        "nivel":
                            nivel_explicito(
                                " | ".join(
                                    linha
                                )
                            ),

                        "controle":
                            controle,

                        "protecao":
                            "",

                        "apoio":
                            ""

                    })

                break


        # ----------------------------------------------------
        # Detecta possíveis tarefas
        # ----------------------------------------------------

        texto_linha = normalizar(
            " ".join(linha)
        )

        palavras_tarefa = [

            "etapa",
            "montagem",
            "instalacao",
            "fixacao",
            "preparacao",
            "acesso",
            "movimentacao",
            "transporte",
            "limpeza",
            "sinalizacao",
            "perfuracao",
            "corte"

        ]

        if any(
            palavra in texto_linha
            for palavra
            in palavras_tarefa
        ):

            ultima_tarefa = (
                " ".join(linha)
            )


    return registros


# ============================================================
# CHUNKS PARA BUSCA
# ============================================================

def criar_chunks(
    texto,
    tamanho=1800,
    sobreposicao=250
):

    if len(texto) <= tamanho:

        return [
            texto
        ]

    resultado = []

    inicio = 0

    while inicio < len(texto):

        fim = (
            inicio
            +
            tamanho
        )

        resultado.append(
            texto[
                inicio:fim
            ]
        )

        inicio = (
            fim
            -
            sobreposicao
        )

    return resultado


# ============================================================
# ÍNDICE SEMÂNTICO
# ============================================================

def preparar_indice(
    base
):

    textos = []
    metadados = []

    for item in base:

        chunks = criar_chunks(
            item["texto"]
        )

        for chunk in chunks:

            textos.append(
                chunk
            )

            metadados.append(
                item
            )


    modelo = carregar_modelo()

    embeddings = modelo.encode(
        textos,
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return (
        np.asarray(
            embeddings,
            dtype=np.float32
        ),
        metadados
    )


# ============================================================
# BUSCA SEMÂNTICA
# ============================================================

def buscar(
    consulta,
    embeddings,
    metadados,
    quantidade
):

    modelo = carregar_modelo()

    query = modelo.encode(
        [consulta],
        normalize_embeddings=True,
        show_progress_bar=False
    )[0]

    similaridades = np.dot(
        embeddings,
        query
    )

    melhores = {}

    for indice, similaridade in enumerate(
        similaridades
    ):

        arquivo = (
            metadados[
                indice
            ]["arquivo"]
        )

        if (
            arquivo not in melhores
            or
            similaridade
            >
            melhores[
                arquivo
            ]
        ):

            melhores[
                arquivo
            ] = float(
                similaridade
            )


    resultados = []

    for arquivo, similaridade in (
        melhores.items()
    ):

        percentual = (
            (
                similaridade
                + 1
            )
            /
            2
        ) * 100

        resultados.append({

            "arquivo":
                arquivo,

            "similaridade":
                percentual

        })


    resultados.sort(
        key=lambda x:
            x["similaridade"],
        reverse=True
    )

    return resultados[
        :quantidade
    ]


# ============================================================
# CONSOLIDAÇÃO ESTRUTURADA
# ============================================================

def consolidar(
    atividade,
    referencias,
    base
):

    arquivos = {
        x["arquivo"]
        for x in referencias
    }

    registros = []

    for item in base:

        if (
            item["arquivo"]
            in arquivos
        ):

            registros.extend(
                item["registros"]
            )


    # Remove registros vazios
    registros = [

        x for x in registros

        if (
            x["risco"]
            and
            x["risco"].strip()
        )

    ]


    # --------------------------------------------------------
    # Similaridade semântica Tarefa + Risco
    # --------------------------------------------------------

    if registros:

        modelo = carregar_modelo()

        textos = [

            (
                x["tarefa"]
                +
                " "
                +
                x["risco"]
            )

            for x in registros

        ]

        embeddings = modelo.encode(

            [atividade]
            +
            textos,

            normalize_embeddings=True,
            show_progress_bar=False

        )

        scores = np.dot(
            embeddings[1:],
            embeddings[0]
        )

        ordem = np.argsort(
            scores
        )[::-1]

        registros = [
            registros[i]
            for i in ordem
        ]


    # --------------------------------------------------------
    # Remover duplicidades
    # --------------------------------------------------------

    resultado = []

    vistos = set()

    for registro in registros:

        chave = (

            normalizar(
                registro["tarefa"]
            ),

            normalizar(
                registro["risco"]
            )

        )

        if chave in vistos:

            continue

        vistos.add(
            chave
        )

        resultado.append(
            registro
        )

        if len(resultado) >= 12:

            break


    return resultado


# ============================================================
# PREENCHER MODELO
# ============================================================

def preencher_modelo(
    modelo_bytes,
    dados,
    registros
):

    workbook = load_workbook(
        io.BytesIO(
            modelo_bytes
        )
    )


    nome_aba = (
        "FORM-ST-0001-APR - Análise Pre"
    )


    if nome_aba not in (
        workbook.sheetnames
    ):

        raise ValueError(
            "Aba principal do modelo "
            "não encontrada."
        )


    ws = workbook[
        nome_aba
    ]


    # --------------------------------------------------------
    # CABEÇALHO
    # --------------------------------------------------------

    ws["A4"] = (
        dados["responsavel_atividade"]
    )

    ws["C4"] = (
        dados["cargo"]
    )

    ws["H4"] = (
        dados["contratante"]
    )

    ws["J4"] = (
        dados["ticket"]
    )

    ws["A6"] = (
        dados["empresa"]
    )

    ws["C6"] = (
        dados["responsavel"]
    )

    ws["H6"] = (
        f"Início: {dados['inicio']} "
        f"                     "
        f"Fim: {dados['fim']}"
    )

    ws["C7"] = (
        dados["atividade"]
    )

    ws["C8"] = (
        dados["data_center"]
    )

    ws["I8"] = (
        dados["localizacao"]
    )


    # --------------------------------------------------------
    # LIMPA APENAS A ÁREA DE RESULTADO
    # --------------------------------------------------------

    for linha in range(
        33,
        70
    ):

        for coluna in [
            1, 3, 7, 8, 9, 10
        ]:

            ws.cell(
                linha,
                coluna
            ).value = None


    # --------------------------------------------------------
    # PREENCHIMENTO
    # --------------------------------------------------------

    for indice, registro in enumerate(
        registros,
        start=33
    ):

        ws.cell(
            indice,
            1
        ).value = (
            registro["tarefa"]
        )

        ws.cell(
            indice,
            3
        ).value = (
            registro["risco"]
        )

        ws.cell(
            indice,
            7
        ).value = (
            registro["nivel"]
            or
            "VALIDAR"
        )

        ws.cell(
            indice,
            8
        ).value = (
            registro["controle"]
            or
            "VALIDAR"
        )

        ws.cell(
            indice,
            9
        ).value = (
            registro["protecao"]
            or
            "VALIDAR"
        )

        ws.cell(
            indice,
            10
        ).value = (
            registro["apoio"]
            or
            "VALIDAR"
        )


        for coluna in [
            1, 3, 7, 8, 9, 10
        ]:

            ws.cell(
                indice,
                coluna
            ).alignment = copy(
                ws.cell(
                    indice,
                    coluna
                ).alignment
            )


    # --------------------------------------------------------
    # SALVAR
    # --------------------------------------------------------

    saida = io.BytesIO()

    workbook.save(
        saida
    )

    saida.seek(
        0
    )

    return saida.getvalue()


# ============================================================
# SESSION STATE
# ============================================================

if "base" not in st.session_state:

    st.session_state.base = []


if "embeddings" not in st.session_state:

    st.session_state.embeddings = None


if "metadados" not in st.session_state:

    st.session_state.metadados = []


if "resultados" not in st.session_state:

    st.session_state.resultados = []


if "arquivo_final" not in st.session_state:

    st.session_state.arquivo_final = None


# ============================================================
# TOPO
# ============================================================

col1, col2, col3 = st.columns(
    3
)

with col1:

    st.metric(
        "APRs",
        "10 APRs"
    )

with col2:

    st.metric(
        "Meta",
        "50 minutos"
    )

with col3:

    st.metric(
        "Meta por APR",
        "≤ 5 minutos"
    )


st.divider()


# ============================================================
# 1 — FONTE
# ============================================================

st.header(
    "📦 1. Carregar base FONTE"
)


fonte = st.file_uploader(
    "Selecione o FONTE.zip",
    type=["zip"]
)


if fonte:

    if st.button(
        "📥 INDEXAR BASE FONTE",
        type="primary"
    ):

        inicio = time.time()

        base = []

        with zipfile.ZipFile(
            fonte
        ) as arquivo_zip:

            arquivos = [

                nome

                for nome
                in arquivo_zip.namelist()

                if nome.lower().endswith(
                    (
                        ".xlsx",
                        ".xlsm"
                    )
                )

                and
                not nome.startswith(
                    "__MACOSX"
                )

            ]


            progresso = st.progress(
                0
            )


            for indice, nome in enumerate(
                arquivos,
                start=1
            ):

                try:

                    base.extend(
                        ler_excel(
                            arquivo_zip.read(
                                nome
                            ),
                            nome
                        )
                    )

                except Exception as erro:

                    st.warning(
                        f"Erro em {nome}: "
                        f"{erro}"
                    )

                progresso.progress(
                    indice
                    /
                    len(arquivos)
                )


        st.session_state.base = (
            base
        )

        st.session_state.embeddings = None

        st.session_state.resultados = []

        quantidade_arquivos = len(
            set(
                x["arquivo"]
                for x in base
            )
        )

        st.success(
            f"Base indexada: "
            f"{quantidade_arquivos} arquivos "
            f"e {len(base)} planilhas "
            f"em "
            f"{time.time() - inicio:.1f} segundos."
        )


# ============================================================
# 2 — SEMÂNTICA
# ============================================================

if st.session_state.base:

    st.divider()

    st.header(
        "🧠 2. Preparar busca semântica"
    )


    col1, col2 = st.columns(
        2
    )


    with col1:

        st.metric(
            "Arquivos Excel",
            len(
                set(
                    x["arquivo"]
                    for x
                    in st.session_state.base
                )
            )
        )


    with col2:

        st.metric(
            "Planilhas indexadas",
            len(
                st.session_state.base
            )
        )


    if st.button(
        "🚀 PREPARAR BUSCA SEMÂNTICA",
        type="primary"
    ):

        inicio = time.time()

        with st.spinner(
            "Criando índice semântico..."
        ):

            (
                embeddings,
                metadados
            ) = preparar_indice(
                st.session_state.base
            )


        st.session_state.embeddings = (
            embeddings
        )

        st.session_state.metadados = (
            metadados
        )


        st.success(
            f"Busca semântica preparada "
            f"em "
            f"{time.time() - inicio:.1f} segundos."
        )


# ============================================================
# 3 — BUSCA
# ============================================================

if (
    st.session_state.embeddings
    is not None
):

    st.divider()

    st.header(
        "🔎 3. Buscar APRs de referência"
    )


    atividade_busca = st.text_input(
        "Descreva a atividade:",
        value=(
            "Montagem e fixação "
            "de eletrocalhas em altura"
        )
    )


    quantidade = st.slider(
        "Quantidade de APRs de referência",
        3,
        10,
        5
    )


    if st.button(
        "🔍 BUSCAR APRs",
        type="primary"
    ):

        inicio = time.time()

        st.session_state.resultados = (
            buscar(
                atividade_busca,
                st.session_state.embeddings,
                st.session_state.metadados,
                quantidade
            )
        )


        st.success(
            f"Busca concluída em "
            f"{time.time() - inicio:.2f} segundos."
        )


# ============================================================
# 4 — RESULTADOS
# ============================================================

if st.session_state.resultados:

    st.divider()

    st.header(
        "🎯 4. APRs de referência"
    )


    opcoes = [

        (
            f"{indice + 1}. "
            f"{resultado['arquivo']} "
            f"— "
            f"{resultado['similaridade']:.1f}%"
        )

        for indice, resultado
        in enumerate(
            st.session_state.resultados
        )

    ]


    selecionadas = st.multiselect(
        "Selecione as APRs de referência:",
        opcoes,
        default=opcoes[
            :min(
                3,
                len(opcoes)
            )
        ]
    )


    for opcao in selecionadas:

        st.write(
            "• " + opcao
        )


    # ========================================================
    # 5 — DADOS
    # ========================================================

    st.divider()

    st.header(
        "📝 5. Dados da nova APR"
    )


    col1, col2 = st.columns(
        2
    )


    with col1:

        responsavel_atividade = st.text_input(
            "Responsável pela atividade"
        )

        cargo = st.text_input(
            "Cargo"
        )

        contratante = st.text_input(
            "Contratante"
        )

        ticket = st.text_input(
            "Nº Ticket"
        )

        empresa = st.text_input(
            "Nome da empresa"
        )


    with col2:

        responsavel = st.text_input(
            "Responsável"
        )

        inicio_execucao = st.text_input(
            "Início"
        )

        fim_execucao = st.text_input(
            "Fim"
        )

        data_center = st.text_input(
            "Data Center"
        )

        localizacao = st.text_input(
            "Localização"
        )


    atividade = st.text_area(
        "Descrição do serviço ou atividade",
        value=atividade_busca
    )


    # ========================================================
    # 6 — MODELO
    # ========================================================

    st.divider()

    st.header(
        "📄 6. Modelo Excel"
    )


    modelo_excel = st.file_uploader(
        "Envie o modelo oficial da APR",
        type=["xlsx"],
        key="modelo_excel"
    )


    # ========================================================
    # 7 — GERAÇÃO
    # ========================================================

    if (
        modelo_excel
        and
        selecionadas
    ):

        st.divider()

        st.header(
            "🚀 7. Gerar APR"
        )


        st.warning(
            "A V7.1 utiliza somente relações "
            "encontradas nas APRs de referência. "
            "Quando a fonte não apresentar "
            "claramente A/M/B ou uma barreira, "
            "o campo será marcado como VALIDAR."
        )


        if st.button(
            "🚀 GERAR APR NO MODELO PADRÃO",
            type="primary"
        ):

            inicio = time.time()


            arquivos_referencia = [

                st.session_state.resultados[
                    opcoes.index(
                        selecionada
                    )
                ]

                for selecionada
                in selecionadas

            ]


            registros = consolidar(
                atividade,
                arquivos_referencia,
                st.session_state.base
            )


            dados = {

                "responsavel_atividade":
                    responsavel_atividade,

                "cargo":
                    cargo,

                "contratante":
                    contratante,

                "ticket":
                    ticket,

                "empresa":
                    empresa,

                "responsavel":
                    responsavel,

                "inicio":
                    inicio_execucao,

                "fim":
                    fim_execucao,

                "atividade":
                    atividade,

                "data_center":
                    data_center,

                "localizacao":
                    localizacao

            }


            arquivo_final = preencher_modelo(
                modelo_excel.getvalue(),
                dados,
                registros
            )


            st.session_state.arquivo_final = (
                arquivo_final
            )


            st.session_state.registros = (
                registros
            )


            nome_saida = (
                "APR_IA_"
                +
                re.sub(
                    r"_+",
                    "_",
                    normalizar(
                        atividade
                    ).replace(
                        " ",
                        "_"
                    )
                )[:70]
                +
                ".xlsx"
            )


            st.session_state.nome_saida = (
                nome_saida
            )


            st.success(
                f"APR estruturada com "
                f"{len(registros)} registros "
                f"em "
                f"{time.time() - inicio:.2f} segundos."
            )


            # =================================================
            # PRÉ-VISUALIZAÇÃO
            # =================================================

            st.subheader(
                "🔎 Pré-visualização antes do Excel"
            )


            preview = []

            for registro in registros:

                preview.append({

                    "Descrição das tarefas":
                        registro["tarefa"],

                    "Riscos associados":
                        registro["risco"],

                    "Nível":
                        registro["nivel"]
                        or
                        "VALIDAR",

                    "Barreira de controle":
                        registro["controle"]
                        or
                        "VALIDAR",

                    "Barreira de proteção":
                        registro["protecao"]
                        or
                        "VALIDAR",

                    "Barreira de apoio":
                        registro["apoio"]
                        or
                        "VALIDAR"

                })


            st.dataframe(
                preview,
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# 8 — DOWNLOAD
# ============================================================

if (
    st.session_state.arquivo_final
    is not None
):

    st.divider()

    st.header(
        "📥 8. APR gerada"
    )


    st.download_button(

        "📥 BAIXAR APR GERADA",

        data=(
            st.session_state.arquivo_final
        ),

        file_name=(
            st.session_state.nome_saida
        ),

        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),

        type="primary"

    )


st.divider()

st.caption(
    "POC Gerador de APR com IA — V7.1 | "
    "Extração estruturada + busca semântica + "
    "modelo oficial"
)

st.caption(
    "⚠️ A saída é preliminar e requer revisão, "
    "validação e aprovação conforme o processo "
    "de Segurança do Trabalho."
)
