import streamlit as st
import zipfile
import io
import re
import time
import numpy as np

from copy import copy
from datetime import datetime
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
    "POC — Busca, consolidação e geração automática de APR"
)

st.info(
    "V7.0 — O sistema encontra APRs de referência, "
    "consolida informações técnicas e gera uma nova APR "
    "utilizando o modelo Excel padrão."
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
    "concretar",
    "inspecao",
    "inspecionar",
    "limpeza",
    "limpar",
    "sinalizacao",
    "sinalizar",
    "posicionamento",
    "posicionar",
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
    "vala",
    "externo",
    "interno"
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
# SCORE
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


def calcular_score_tecnico(
    consulta,
    texto
):

    grupos = classificar_consulta(
        consulta
    )

    score_objeto, objetos = (
        calcular_score_grupo(
            grupos["objetos"],
            texto
        )
    )

    score_atividade, atividades = (
        calcular_score_grupo(
            grupos["atividades"],
            texto
        )
    )

    score_contexto, contextos = (
        calcular_score_grupo(
            grupos["contextos"],
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
        objetos,
        atividades,
        contextos,
        min(score_objeto, 100)
    )


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

    score_objeto, _ = (
        calcular_score_grupo(
            grupos["objetos"],
            titulo
        )
    )

    score_atividade, _ = (
        calcular_score_grupo(
            grupos["atividades"],
            titulo
        )
    )

    score_contexto, _ = (
        calcular_score_grupo(
            grupos["contextos"],
            titulo
        )
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
# ÍNDICE SEMÂNTICO
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
# BUSCA
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

    melhores = {}

    for indice, similaridade in enumerate(
        similaridades
    ):

        arquivo = metadados[
            indice
        ]["arquivo"]

        if (
            arquivo not in melhores
            or
            similaridade >
            melhores[
                arquivo
            ]["similaridade"]
        ):

            melhores[
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

    arquivos_processados = set()

    for item in base_aprs:

        arquivo = item[
            "arquivo"
        ]

        if arquivo in arquivos_processados:
            continue

        arquivos_processados.add(
            arquivo
        )

        if arquivo not in melhores:
            continue

        similaridade = (
            melhores[
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
                * 100
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

    return resultados[:quantidade]


# ============================================================
# CANDIDATOS PARA GERAÇÃO
# ============================================================

TERMOS_TAREFA = {
    "montagem",
    "instalação",
    "instalacao",
    "fixação",
    "fixacao",
    "inspeção",
    "inspecao",
    "preparação",
    "preparacao",
    "acesso",
    "movimentação",
    "movimentacao",
    "transporte",
    "corte",
    "perfuração",
    "perfuracao",
    "passagem",
    "lançamento",
    "lancamento",
    "posicionamento",
    "limpeza",
    "desmontagem",
    "sinalização",
    "sinalizacao",
    "utilização",
    "utilizacao",
    "execução",
    "execucao"
}


TERMOS_RISCO = {
    "queda",
    "choque",
    "elétrico",
    "eletrico",
    "corte",
    "prensamento",
    "esmagamento",
    "atropelamento",
    "colisão",
    "colisao",
    "projeção",
    "projecao",
    "incêndio",
    "incendio",
    "explosão",
    "explosao",
    "ruído",
    "ruido",
    "vibração",
    "vibracao",
    "ergonômico",
    "ergonomico",
    "exposição",
    "exposicao",
    "material",
    "ferramenta",
    "objetos",
    "lesão",
    "lesao",
    "acidente"
}


TERMOS_CONTROLE = {
    "inspeção",
    "inspecao",
    "sinalização",
    "sinalizacao",
    "isolamento",
    "bloqueio",
    "desenergização",
    "desenergizacao",
    "autorização",
    "autorizacao",
    "amarração",
    "amarracao",
    "delimitação",
    "delimitacao",
    "proteção",
    "protecao",
    "guarda-corpo",
    "guardacorpo",
    "linha de vida",
    "checklist",
    "controle",
    "verificação",
    "verificacao",
    "permissão",
    "permissao",
    "organização",
    "organizacao"
}


TERMOS_PROTECAO = {
    "capacete",
    "óculos",
    "oculos",
    "luva",
    "luvas",
    "calçado",
    "calcado",
    "cinto",
    "talabarte",
    "protetor",
    "respirador",
    "máscara",
    "mascara",
    "epi",
    "equipamento de proteção",
    "equipamento de protecao"
}


TERMOS_APOIO = {
    "nr-",
    "nr ",
    "procedimento",
    "treinamento",
    "capacitação",
    "capacitacao",
    "supervisão",
    "supervisao",
    "autorização",
    "autorizacao",
    "apr",
    "instrução",
    "instrucao",
    "certificação",
    "certificacao",
    "permissão",
    "permissao"
}


def linha_relevante(
    linha
):

    texto = normalizar(
        linha
    )

    if len(texto) < 12:
        return False

    return any(
        termo in texto
        for termo in TERMOS_TAREFA
    )


def extrair_candidatos(
    textos
):

    tarefas = []
    riscos = []
    controles = []
    protecoes = []
    apoios = []

    for texto in textos:

        linhas = [
            x.strip()
            for x in str(texto).splitlines()
            if x.strip()
        ]

        for linha in linhas:

            if len(linha) > 500:
                continue

            normalizada = normalizar(
                linha
            )

            if linha_relevante(
                linha
            ):

                tarefas.append(
                    linha
                )

            if any(
                termo in normalizada
                for termo in TERMOS_RISCO
            ):

                riscos.append(
                    linha
                )

            if any(
                termo in normalizada
                for termo in TERMOS_CONTROLE
            ):

                controles.append(
                    linha
                )

            if any(
                termo in normalizada
                for termo in TERMOS_PROTECAO
            ):

                protecoes.append(
                    linha
                )

            if any(
                termo in normalizada
                for termo in TERMOS_APOIO
            ):

                apoios.append(
                    linha
                )

    return {
        "tarefas": remover_duplicados(
            tarefas
        ),
        "riscos": remover_duplicados(
            riscos
        ),
        "controles": remover_duplicados(
            controles
        ),
        "protecoes": remover_duplicados(
            protecoes
        ),
        "apoios": remover_duplicados(
            apoios
        )
    }


def remover_duplicados(
    itens
):

    resultado = []
    vistos = set()

    for item in itens:

        chave = normalizar(
            item
        )

        if chave in vistos:
            continue

        vistos.add(
            chave
        )

        resultado.append(
            item
        )

    return resultado


# ============================================================
# SIMILARIDADE ENTRE LINHAS
# ============================================================

def selecionar_semelhantes(
    consulta,
    candidatos,
    modelo,
    limite=2
):

    if not candidatos:

        return []

    textos = (
        [consulta]
        +
        candidatos
    )

    embeddings = modelo.encode(
        textos,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    query = embeddings[0]

    scores = np.dot(
        embeddings[1:],
        query
    )

    pares = list(
        zip(
            candidatos,
            scores
        )
    )

    pares.sort(
        key=lambda x: x[1],
        reverse=True
    )

    resultado = []

    for texto, score in pares[:limite]:

        resultado.append(
            texto
        )

    return resultado


# ============================================================
# NÍVEL DE RISCO
# ============================================================

def identificar_nivel(
    texto_risco
):

    texto = normalizar(
        texto_risco
    )

    # Primeiro tenta encontrar classificação
    # explicitamente registrada na fonte.

    padroes = [
        r"\|\s*([AMB])\s*\|",
        r"\b(?:nivel|nível)\s*[:\-]?\s*([AMB])\b",
        r"\b([AMB])\s*[-–]\s*(?:alto|medio|médio|baixo)\b"
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


    # Heurística conservadora apenas quando
    # a fonte não apresenta a classificação.

    if any(
        termo in texto
        for termo in [
            "queda de altura",
            "choque eletrico",
            "choque elétrico",
            "atropelamento",
            "esmagamento",
            "explosao",
            "explosão"
        ]
    ):

        return "A"


    if any(
        termo in texto
        for termo in [
            "corte",
            "projecao",
            "projeção",
            "queda de material",
            "ruido",
            "ruído"
        ]
    ):

        return "M"


    return "B"


# ============================================================
# CONSOLIDAÇÃO
# ============================================================

def consolidar_dados(
    resultados,
    modelo,
    max_linhas=16
):

    textos = [
        resultado["texto"]
        for resultado
        in resultados
    ]

    candidatos = extrair_candidatos(
        textos
    )

    if not candidatos["tarefas"]:

        # Fallback:
        # utiliza linhas mais longas do documento.

        for texto in textos:

            for linha in str(texto).splitlines():

                linha = linha.strip()

                if (
                    25
                    <= len(linha)
                    <= 300
                ):

                    candidatos[
                        "tarefas"
                    ].append(
                        linha
                    )

        candidatos["tarefas"] = (
            remover_duplicados(
                candidatos["tarefas"]
            )
        )


    tarefas = candidatos[
        "tarefas"
    ][:max_linhas]


    linhas = []


    for tarefa in tarefas:

        riscos = selecionar_semelhantes(
            tarefa,
            candidatos["riscos"],
            modelo,
            limite=2
        )

        controles = selecionar_semelhantes(
            tarefa,
            candidatos["controles"],
            modelo,
            limite=2
        )

        protecoes = selecionar_semelhantes(
            tarefa,
            candidatos["protecoes"],
            modelo,
            limite=2
        )

        apoios = selecionar_semelhantes(
            tarefa,
            candidatos["apoios"],
            modelo,
            limite=2
        )


        risco_texto = (
            "\n".join(
                [
                    f"{i + 1}. {x}"
                    for i, x
                    in enumerate(riscos)
                ]
            )
        )


        controle_texto = (
            "\n".join(
                [
                    f"{i + 1}. {x}"
                    for i, x
                    in enumerate(controles)
                ]
            )
        )


        protecao_texto = (
            "\n".join(
                [
                    f"{i + 1}. {x}"
                    for i, x
                    in enumerate(protecoes)
                ]
            )
        )


        apoio_texto = (
            "\n".join(
                [
                    f"{i + 1}. {x}"
                    for i, x
                    in enumerate(apoios)
                ]
            )
        )


        nivel = identificar_nivel(
            risco_texto
        )


        linhas.append({

            "tarefa":
                tarefa,

            "risco":
                risco_texto
                or
                "Validar riscos da tarefa",

            "nivel":
                nivel,

            "controle":
                controle_texto
                or
                "Validar barreiras de controle",

            "protecao":
                protecao_texto
                or
                "Validar barreiras de proteção",

            "apoio":
                apoio_texto
                or
                "Validar procedimentos, treinamento e supervisão"

        })


    return linhas


# ============================================================
# MARCAÇÃO DE ATIVIDADE DE ALTO RISCO
# ============================================================

def detectar_alto_risco(
    atividade
):

    texto = normalizar(
        atividade
    )

    riscos = {

        "altura":
            "altura" in texto,

        "confinado":
            "confinado" in texto,

        "materiais":
            "material" in texto
            or
            "eletrocalha" in texto,

        "perigosa":
            "substancia perigosa" in texto,

        "quente":
            "quente" in texto
            or
            "soldagem" in texto,

        "rua":
            "rua" in texto,

        "carga":
            "içamento" in texto
            or
            "icamento" in texto,

        "eletrico":
            "eletrico" in texto
            or
            "elétrica" in texto
            or
            "eletricas" in texto,

        "maquinas":
            "maquinas pesadas" in texto,

        "escavacao":
            "escavacao" in texto,

        "loto":
            "loto" in texto

    }

    return riscos


# ============================================================
# PREENCHER MODELO EXCEL
# ============================================================

def preencher_modelo_excel(
    template_bytes,
    dados,
    linhas_apr,
    nome_saida
):

    workbook = load_workbook(
        io.BytesIO(
            template_bytes
        )
    )


    nome_aba = (
        "FORM-ST-0001-APR - Análise Pre"
    )


    if nome_aba not in workbook.sheetnames:

        raise ValueError(
            "A aba principal do modelo "
            "não foi encontrada."
        )


    ws = workbook[
        nome_aba
    ]


    # --------------------------------------------------------
    # CABEÇALHO
    # --------------------------------------------------------

    ws["A4"] = dados.get(
        "responsavel_atividade",
        ""
    )

    ws["C4"] = dados.get(
        "cargo",
        ""
    )

    ws["H4"] = dados.get(
        "contratante",
        ""
    )

    ws["J4"] = dados.get(
        "ticket",
        ""
    )

    ws["A6"] = dados.get(
        "empresa",
        ""
    )

    ws["C6"] = dados.get(
        "responsavel",
        ""
    )

    ws["H6"] = (
        f"Inicio: "
        f"{dados.get('inicio', '')}"
        f"                         "
        f"Fim: "
        f"{dados.get('fim', '')}"
    )

    ws["C7"] = dados.get(
        "atividade",
        ""
    )

    ws["C8"] = dados.get(
        "data_center",
        ""
    )

    ws["I8"] = dados.get(
        "localizacao",
        ""
    )


    # --------------------------------------------------------
    # ATIVIDADES DE ALTO RISCO
    # --------------------------------------------------------

    alto_risco = detectar_alto_risco(
        dados.get(
            "atividade",
            ""
        )
    )


    def marcar(texto, chave):

        if alto_risco.get(
            chave,
            False
        ):

            return re.sub(
                r"\(\s*\)",
                "( X )",
                texto,
                count=1
            )

        return texto


    ws["A10"] = marcar(
        ws["A10"].value,
        "altura"
    )

    ws["E10"] = marcar(
        ws["E10"].value,
        "confinado"
    )

    ws["I10"] = marcar(
        ws["I10"].value,
        "materiais"
    )

    ws["A11"] = marcar(
        ws["A11"].value,
        "perigosa"
    )

    ws["E11"] = marcar(
        ws["E11"].value,
        "quente"
    )

    ws["I11"] = marcar(
        ws["I11"].value,
        "rua"
    )

    ws["A12"] = marcar(
        ws["A12"].value,
        "carga"
    )

    ws["E12"] = marcar(
        ws["E12"].value,
        "eletrico"
    )

    ws["I12"] = marcar(
        ws["I12"].value,
        "maquinas"
    )

    ws["A13"] = marcar(
        ws["A13"].value,
        "loto"
    )

    ws["E13"] = marcar(
        ws["E13"].value,
        "escavacao"
    )


    # --------------------------------------------------------
    # LIMPAR ÁREA DA APR
    # --------------------------------------------------------

    for linha in range(
        33,
        49
    ):

        ws.cell(
            linha,
            1
        ).value = None

        ws.cell(
            linha,
            3
        ).value = None

        ws.cell(
            linha,
            7
        ).value = None

        ws.cell(
            linha,
            8
        ).value = None

        ws.cell(
            linha,
            9
        ).value = None

        ws.cell(
            linha,
            10
        ).value = None


    # --------------------------------------------------------
    # PREENCHER LINHAS
    # --------------------------------------------------------

    for indice, item in enumerate(
        linhas_apr[:16],
        start=33
    ):

        ws.cell(
            indice,
            1
        ).value = item[
            "tarefa"
        ]

        ws.cell(
            indice,
            3
        ).value = item[
            "risco"
        ]

        ws.cell(
            indice,
            7
        ).value = item[
            "nivel"
        ]

        ws.cell(
            indice,
            8
        ).value = item[
            "controle"
        ]

        ws.cell(
            indice,
            9
        ).value = item[
            "protecao"
        ]

        ws.cell(
            indice,
            10
        ).value = item[
            "apoio"
        ]


        # Preserva estilos já existentes
        # e ajusta alinhamento.

        for coluna in [
            1, 3, 8, 9, 10
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
    # NOME DO ARQUIVO
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

if "linhas_apr" not in st.session_state:
    st.session_state.linhas_apr = []


# ============================================================
# 1. BASE FONTE
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

        st.session_state.linhas_apr = []


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

                total = len(
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

                        dados_excel = ler_excel(
                            zip_ref.read(nome),
                            nome
                        )

                        base_aprs.extend(
                            dados_excel
                        )

                    except Exception as e:

                        st.warning(
                            f"Erro em "
                            f"{nome}: {e}"
                        )

                    progresso.progress(
                        contador / total
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

        tempo = (
            time.time()
            -
            inicio
        )

        quantidade = len(
            set(
                x["arquivo"]
                for x in base_aprs
            )
        )

        st.success(
            f"Base indexada: "
            f"{quantidade} arquivos e "
            f"{len(base_aprs)} planilhas "
            f"em {tempo:.1f} segundos."
        )


# ============================================================
# 2. MODELO SEMÂNTICO
# ============================================================

if st.session_state.base_aprs:

    st.divider()

    st.header(
        "🧠 2. Preparar busca semântica"
    )

    total_aprs = len(
        set(
            x["arquivo"]
            for x in st.session_state.base_aprs
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
                "Carregando modelo..."
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
                "Erro ao preparar o modelo."
            )

            st.exception(e)


# ============================================================
# 3. CONSULTA
# ============================================================

if st.session_state.indice_pronto:

    st.divider()

    st.header(
        "🔎 3. Descrever a atividade"
    )

    consulta = st.text_input(
        "Atividade que deseja gerar:",
        placeholder=(
            "Ex.: Montagem e fixação "
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
        "🔍 BUSCAR APRs DE REFERÊNCIA",
        type="primary"
    ):

        if not consulta.strip():

            st.warning(
                "Informe a atividade."
            )

        else:

            inicio = time.time()

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

            tempo = (
                time.time()
                -
                inicio
            )

            st.success(
                f"{len(resultados)} APRs encontradas "
                f"em {tempo:.2f} segundos."
            )


# ============================================================
# 4. SELEÇÃO DAS APRs
# ============================================================

if st.session_state.resultados_busca:

    st.divider()

    st.header(
        "🎯 4. Selecionar APRs de referência"
    )

    opcoes = [

        f"{i + 1}. "
        f"{r['arquivo']} "
        f"— {r['score_hibrido']:.1f}%"

        for i, r
        in enumerate(
            st.session_state.resultados_busca
        )

    ]


    selecionadas = st.multiselect(
        "Selecione as APRs que servirão como referência técnica:",
        options=opcoes,
        default=opcoes[:min(3, len(opcoes))]
    )


    if selecionadas:

        st.info(
            f"{len(selecionadas)} APR(s) "
            "selecionada(s) para consolidação."
        )


        for opcao in selecionadas:

            indice = opcoes.index(
                opcao
            )

            resultado = (
                st.session_state.resultados_busca[
                    indice
                ]
            )

            st.write(
                f"• **{resultado['arquivo']}** "
                f"— índice "
                f"{resultado['score_hibrido']:.1f}%"
            )


# ============================================================
# 5. DADOS DA NOVA APR
# ============================================================

if st.session_state.resultados_busca:

    st.divider()

    st.header(
        "📝 5. Dados da nova APR"
    )

    col1, col2 = st.columns(2)

    with col1:

        responsavel_atividade = st.text_input(
            "Responsável pela atividade",
            ""
        )

        cargo = st.text_input(
            "Cargo",
            ""
        )

        contratante = st.text_input(
            "Contratante",
            ""
        )

        ticket = st.text_input(
            "Nº Ticket",
            ""
        )

        empresa = st.text_input(
            "Nome da empresa contratante",
            ""
        )


    with col2:

        responsavel = st.text_input(
            "Responsável",
            ""
        )

        inicio_execucao = st.text_input(
            "Início",
            ""
        )

        fim_execucao = st.text_input(
            "Fim",
            ""
        )

        data_center = st.text_input(
            "Data Center",
            ""
        )

        localizacao = st.text_input(
            "Localização",
            ""
        )


    atividade = st.text_area(
        "Descrição do serviço ou atividade",
        value="",
        height=80
    )


# ============================================================
# 6. MODELO EXCEL
# ============================================================

if st.session_state.resultados_busca:

    st.divider()

    st.header(
        "📄 6. Modelo Excel de saída"
    )

    template_file = st.file_uploader(
        "Envie o modelo oficial da APR",
        type=["xlsx"],
        key="template_apr"
    )


    if template_file:

        st.success(
            f"Modelo carregado: "
            f"{template_file.name}"
        )

        st.caption(
            "O sistema utilizará esse arquivo como "
            "template e preservará as demais abas "
            "do modelo."
        )


# ============================================================
# 7. GERAR APR
# ============================================================

if (
    st.session_state.resultados_busca
    and
    template_file is not None
    and
    selecionadas
):

    st.divider()

    st.header(
        "🚀 7. Gerar APR"
    )

    st.warning(
        "A APR gerada nesta V7.0 é uma versão "
        "preliminar assistida, baseada nas APRs "
        "selecionadas. Deve ser revisada e aprovada "
        "por profissional responsável antes da execução."
    )


    if st.button(
        "🚀 GERAR APR NO MODELO PADRÃO",
        type="primary"
    ):

        if not atividade.strip():

            st.error(
                "Informe a descrição "
                "da atividade."
            )

        else:

            inicio_geracao = time.time()


            modelo = carregar_modelo()


            # ------------------------------------------------
            # RECUPERAR APRs SELECIONADAS
            # ------------------------------------------------

            resultados_selecionados = []


            for opcao in selecionadas:

                indice = opcoes.index(
                    opcao
                )

                resultados_selecionados.append(
                    st.session_state.resultados_busca[
                        indice
                    ]
                )


            # ------------------------------------------------
            # CONSOLIDAÇÃO
            # ------------------------------------------------

            with st.spinner(
                "Consolidando tarefas, riscos "
                "e barreiras das APRs..."
            ):

                linhas_apr = consolidar_dados(
                    resultados_selecionados,
                    modelo,
                    max_linhas=16
                )


            st.session_state.linhas_apr = (
                linhas_apr
            )


            # ------------------------------------------------
            # DADOS
            # ------------------------------------------------

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


            # ------------------------------------------------
            # NOME
            # ------------------------------------------------

            nome_limpo = normalizar(
                atividade
            )

            nome_limpo = (
                re.sub(
                    r"\s+",
                    "_",
                    nome_limpo
                )[:70]
            )


            nome_saida = (
                f"APR_IA_"
                f"{nome_limpo}.xlsx"
            )


            # ------------------------------------------------
            # GERAR EXCEL
            # ------------------------------------------------

            with st.spinner(
                "Preenchendo o modelo Excel..."
            ):

                arquivo_final = (
                    preencher_modelo_excel(
                        template_file.getvalue(),
                        dados,
                        linhas_apr,
                        nome_saida
                    )
                )


            tempo_total = (
                time.time()
                -
                inicio_geracao
            )


            st.success(
                f"APR gerada em "
                f"{tempo_total:.2f} segundos."
            )


            st.session_state.arquivo_final = (
                arquivo_final
            )

            st.session_state.nome_saida = (
                nome_saida
            )


# ============================================================
# 8. RESULTADO
# ============================================================

if (
    "arquivo_final"
    in st.session_state
):

    st.divider()

    st.header(
        "📥 8. APR gerada"
    )


    st.success(
        "A APR foi criada utilizando "
        "o modelo Excel fornecido."
    )


    if st.session_state.linhas_apr:

        st.subheader(
            "Pré-visualização da matriz gerada"
        )


        dados_preview = []

        for item in (
            st.session_state.linhas_apr
        ):

            dados_preview.append({

                "Descrição das tarefas":
                    item["tarefa"],

                "Riscos":
                    item["risco"],

                "Nível":
                    item["nivel"],

                "Barreira de controle":
                    item["controle"],

                "Barreira de proteção":
                    item["protecao"],

                "Barreira de apoio":
                    item["apoio"]

            })


        st.dataframe(
            dados_preview,
            use_container_width=True,
            hide_index=True
        )


    st.download_button(
        label="📥 BAIXAR APR GERADA",
        data=st.session_state.arquivo_final,
        file_name=st.session_state.nome_saida,
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        type="primary"
    )


# ============================================================
# RODAPÉ
# ============================================================

st.divider()

st.caption(
    "POC Gerador de APR com IA — V7.0 | "
    "Busca + Consolidação + Geração no Modelo Padrão"
)

st.caption(
    "⚠️ A saída é preliminar e requer revisão, "
    "validação e aprovação conforme o processo "
    "de Segurança do Trabalho."
)
