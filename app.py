import streamlit as st
import zipfile
import io
import re
import time
import unicodedata
from collections import Counter
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
st.subheader("POC — Consulta contextual à base de APRs")

st.info(
    "V4.1 — O sistema analisa o conteúdo completo de cada APR "
    "para identificar atividades relacionadas."
)

# ============================================================
# INDICADORES
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🎯 Meta", "10 APRs")

with col2:
    st.metric("⏱️ Meta", "50 minutos")

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
    "pelo", "pelas", "pelos", "uma", "durante",
    "realizar", "realizacao", "execucao",
    "atividade", "servico", "serviços"
}

# ============================================================
# TERMOS GENÉRICOS
# ============================================================

TERMOS_GENERICOS = {
    "instalacao",
    "montagem",
    "montar",
    "servico",
    "servicos",
    "execucao",
    "atividade",
    "trabalho",
    "sistema",
    "processo",
    "realizacao",
    "manutencao",
    "operacao",
    "estrutura",
    "equipamento"
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
    )

    return texto.strip()


def obter_tokens(texto):

    texto = normalizar(texto)

    return [
        palavra
        for palavra in texto.split()
        if len(palavra) >= 3
        and palavra not in STOPWORDS
    ]


# ============================================================
# PESO DOS TERMOS
# ============================================================

def peso_termo(termo):

    if termo in TERMOS_GENERICOS:
        return 0.35

    if len(termo) >= 10:
        return 1.50

    if len(termo) >= 7:
        return 1.30

    if len(termo) >= 5:
        return 1.10

    return 1.00


# ============================================================
# SCORE DE UMA APR COMPLETA
# ============================================================

def calcular_score_apr(consulta, texto_apr):

    consulta_normalizada = normalizar(
        consulta
    )

    texto_normalizado = normalizar(
        texto_apr
    )

    consulta_tokens = obter_tokens(
        consulta
    )

    texto_tokens = obter_tokens(
        texto_apr
    )

    if not consulta_tokens or not texto_tokens:
        return 0, []

    consulta_unica = list(
        dict.fromkeys(
            consulta_tokens
        )
    )

    contador_texto = Counter(
        texto_tokens
    )

    # --------------------------------------------------------
    # 1. Cobertura ponderada
    # --------------------------------------------------------

    peso_total = 0
    peso_encontrado = 0

    termos_encontrados = []

    for termo in consulta_unica:

        peso = peso_termo(termo)

        peso_total += peso

        if contador_texto[termo] > 0:

            peso_encontrado += peso

            termos_encontrados.append(
                termo
            )

    if peso_total == 0:
        return 0, termos_encontrados

    cobertura = (
        peso_encontrado /
        peso_total
    )

    # --------------------------------------------------------
    # 2. Frequência dos termos específicos
    # --------------------------------------------------------

    frequencia_bonus = 0

    for termo in termos_encontrados:

        if termo not in TERMOS_GENERICOS:

            ocorrencias = contador_texto[
                termo
            ]

            frequencia_bonus += min(
                ocorrencias * 1.5,
                8
            )

    frequencia_bonus = min(
        frequencia_bonus,
        15
    )

    # --------------------------------------------------------
    # 3. Frase exata
    # --------------------------------------------------------

    frase_bonus = 0

    if (
        len(consulta_normalizada) >= 10
        and consulta_normalizada in texto_normalizado
    ):

        frase_bonus = 20

    # --------------------------------------------------------
    # 4. Combinação de termos específicos
    # --------------------------------------------------------

    termos_especificos = [
        termo
        for termo in termos_encontrados
        if termo not in TERMOS_GENERICOS
    ]

    combinacao_bonus = 0

    if len(termos_especificos) >= 2:
        combinacao_bonus = 10

    if len(termos_especificos) >= 3:
        combinacao_bonus = 15

    if len(termos_especificos) >= 4:
        combinacao_bonus = 20

    # --------------------------------------------------------
    # 5. Score final
    # --------------------------------------------------------

    score = (
        cobertura * 55
        + frequencia_bonus
        + frase_bonus
        + combinacao_bonus
    )

    score = min(
        score,
        100
    )

    return score, termos_encontrados


# ============================================================
# LEITURA DE UMA APR
# ============================================================

def ler_excel(conteudo, nome_arquivo):

    registros = []

    try:

        arquivo = io.BytesIO(
            conteudo
        )

        workbook = load_workbook(
            arquivo,
            read_only=True,
            data_only=True
        )

        for nome_planilha in workbook.sheetnames:

            planilha = workbook[
                nome_planilha
            ]

            linhas = []

            numero_linha = 0

            for linha in planilha.iter_rows(
                values_only=True
            ):

                numero_linha += 1

                valores = []

                for valor in linha:

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

            texto_planilha = "\n".join(
                linhas
            )

            if texto_planilha.strip():

                registros.append({

                    "arquivo":
                        nome_arquivo,

                    "planilha":
                        nome_planilha,

                    "texto":
                        texto_planilha,

                    "linhas":
                        len(linhas)
                })

        workbook.close()

    except Exception as erro:

        registros.append({

            "arquivo":
                nome_arquivo,

            "planilha":
                "ERRO",

            "texto":
                str(erro),

            "linhas":
                0
        })

    return registros


# ============================================================
# UPLOAD
# ============================================================

st.header(
    "📦 1. Carregar base FONTE"
)

st.write(
    "Envie novamente o arquivo FONTE.zip."
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
        f"Arquivo recebido: "
        f"**{arquivo_zip.name}** "
        f"({tamanho_mb:.1f} MB)"
    )

    if st.button(
        "🔎 INDEXAR BASE DE APRs",
        type="primary"
    ):

        inicio = time.time()

        base_aprs = []

        progresso = st.progress(
            0
        )

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

                for contador, nome in enumerate(
                    arquivos_excel,
                    start=1
                ):

                    status.text(
                        f"Lendo {contador} "
                        f"de {total}: "
                        f"{nome}"
                    )

                    conteudo = zip_ref.read(
                        nome
                    )

                    dados = ler_excel(
                        conteudo,
                        nome
                    )

                    base_aprs.extend(
                        dados
                    )

                    progresso.progress(
                        contador / total
                    )

            tempo = (
                time.time()
                - inicio
            )

            st.session_state[
                "base_aprs"
            ] = base_aprs

            st.session_state[
                "quantidade_arquivos"
            ] = total

            st.session_state[
                "tempo_indexacao"
            ] = tempo

            status.empty()

            st.success(
                f"✅ Base indexada em "
                f"**{tempo:.1f} segundos**."
            )

            col1, col2, col3 = st.columns(
                3
            )

            with col1:

                st.metric(
                    "Arquivos Excel",
                    total
                )

            with col2:

                st.metric(
                    "Planilhas indexadas",
                    len(base_aprs)
                )

            with col3:

                st.metric(
                    "Tempo",
                    f"{tempo:.1f}s"
                )

        except Exception as erro:

            st.error(
                f"Erro durante a indexação: "
                f"{erro}"
            )


# ============================================================
# PESQUISA
# ============================================================

if "base_aprs" in st.session_state:

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

    quantidade = st.slider(
        "Quantidade de APRs semelhantes:",
        1,
        10,
        5
    )

    if st.button(
        "🔍 PESQUISAR APRs",
        type="primary"
    ):

        if not atividade.strip():

            st.warning(
                "Digite uma atividade."
            )

        else:

            inicio = time.time()

            resultados = []

            for apr in st.session_state[
                "base_aprs"
            ]:

                score, termos = (
                    calcular_score_apr(
                        atividade,
                        apr["texto"]
                    )
                )

                if score > 0:

                    resultados.append({

                        "arquivo":
                            apr["arquivo"],

                        "planilha":
                            apr["planilha"],

                        "score":
                            score,

                        "termos":
                            termos,

                        "texto":
                            apr["texto"],

                        "linhas":
                            apr["linhas"]
                    })

            resultados.sort(
                key=lambda x:
                    x["score"],
                reverse=True
            )

            # ------------------------------------------------
            # Remover duplicidades do mesmo arquivo
            # ------------------------------------------------

            resultados_unicos = []

            arquivos_vistos = set()

            for resultado in resultados:

                if resultado[
                    "arquivo"
                ] not in arquivos_vistos:

                    resultados_unicos.append(
                        resultado
                    )

                    arquivos_vistos.add(
                        resultado["arquivo"]
                    )

                if len(
                    resultados_unicos
                ) >= quantidade:

                    break

            tempo = (
                time.time()
                - inicio
            )

            st.success(
                f"Pesquisa concluída em "
                f"**{tempo:.3f} segundos**."
            )

            if not resultados_unicos:

                st.warning(
                    "Nenhuma APR encontrada."
                )

            else:

                st.subheader(
                    f"📋 "
                    f"{len(resultados_unicos)} "
                    f"APR(s) encontrada(s)"
                )

                for numero, resultado in enumerate(
                    resultados_unicos,
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
                            f"**Relevância:** "
                            f"{resultado['score']:.1f}%"
                        )

                        st.write(
                            f"**Linhas analisadas:** "
                            f"{resultado['linhas']}"
                        )

                        st.write(
                            "**Termos encontrados:**"
                        )

                        st.info(
                            ", ".join(
                                resultado[
                                    "termos"
                                ]
                            )
                        )

                        st.write(
                            "**Trecho da APR:**"
                        )

                        st.text_area(
                            "Conteúdo",
                            resultado[
                                "texto"
                            ][:5000],
                            height=250,
                            key=(
                                f"apr_{numero}"
                            )
                        )


# ============================================================
# ROADMAP
# ============================================================

st.divider()

st.header(
    "🚀 Evolução do POC"
)

st.write(
    """
**V4.1 — Busca contextual por APR**

Próximas etapas:

**V5** → Busca semântica com IA

**V6** → Identificação automática de perigos,
riscos, consequências e controles

**V7** → Geração automática da APR

**V8** → Exportação em Excel mantendo o
layout padrão

**Validação final** → 10 APRs em até 50 minutos.
"""
)

st.caption(
    "O percentual apresentado é um índice de "
    "relevância do POC e não representa probabilidade "
    "ou garantia de similaridade técnica."
)
