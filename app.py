import streamlit as st
import zipfile
import io
import re
import time
import unicodedata
from openpyxl import load_workbook

st.set_page_config(
    page_title="Gerador de APR com IA",
    page_icon="🦺",
    layout="wide"
)

# ============================================================
# CONFIGURAÇÃO
# ============================================================

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
# FUNÇÕES
# ============================================================

STOPWORDS = {
    "a", "o", "e", "de", "da", "do", "das", "dos",
    "em", "no", "na", "nos", "nas", "para", "por",
    "com", "sem", "um", "uma", "uns", "umas",
    "ao", "aos", "as", "os", "que", "se", "é",
    "ser", "foi", "são", "como"
}


def normalizar(texto):
    """Remove acentos, pontuação e transforma em minúsculas."""

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


def palavras(texto):
    """Transforma texto em conjunto de palavras relevantes."""

    texto = normalizar(texto)

    return {
        palavra
        for palavra in texto.split()
        if len(palavra) > 2
        and palavra not in STOPWORDS
    }


def similaridade(consulta, texto):
    """Calcula uma similaridade simples entre consulta e APR."""

    palavras_consulta = palavras(consulta)
    palavras_texto = palavras(texto)

    if not palavras_consulta or not palavras_texto:
        return 0

    intersecao = palavras_consulta.intersection(
        palavras_texto
    )

    return len(intersecao) / len(palavras_consulta) * 100


def ler_excel(conteudo, nome_arquivo):
    """Lê todas as planilhas Excel dentro do arquivo."""

    resultados = []

    try:

        arquivo = io.BytesIO(conteudo)

        workbook = load_workbook(
            arquivo,
            read_only=True,
            data_only=True
        )

        for nome_planilha in workbook.sheetnames:

            planilha = workbook[nome_planilha]

            textos = []

            for linha in planilha.iter_rows(
                values_only=True
            ):

                valores = []

                for valor in linha:

                    if valor is not None:

                        valores.append(
                            str(valor)
                        )

                if valores:

                    textos.append(
                        " | ".join(valores)
                    )

            texto_final = "\n".join(textos)

            if texto_final.strip():

                resultados.append({
                    "arquivo": nome_arquivo,
                    "planilha": nome_planilha,
                    "texto": texto_final
                })

        workbook.close()

    except Exception as erro:

        resultados.append({
            "arquivo": nome_arquivo,
            "planilha": "ERRO",
            "texto": f"Não foi possível ler o arquivo: {erro}"
        })

    return resultados


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

    tamanho_mb = arquivo_zip.size / (1024 * 1024)

    st.success(
        f"Arquivo recebido: **{arquivo_zip.name}** "
        f"({tamanho_mb:.1f} MB)"
    )

    if st.button(
        "🔎 INDEXAR BASE DE APRs",
        type="primary"
    ):

        inicio = time.time()

        base = []

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
                    and not nome.startswith("__MACOSX")
                ]

                total = len(arquivos_excel)

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

                        base.extend(dados)

                        progresso.progress(
                            contador / total
                        )

            tempo = time.time() - inicio

            st.session_state["base_aprs"] = base
            st.session_state["quantidade_arquivos"] = total
            st.session_state["tempo_indexacao"] = tempo

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
                    "Planilhas indexadas",
                    len(base)
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

if "base_aprs" in st.session_state:

    st.divider()

    st.header("🔎 2. Pesquisar na base de APRs")

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

            resultados = []

            for item in st.session_state["base_aprs"]:

                score = similaridade(
                    atividade,
                    item["texto"]
                )

                if score > 0:

                    resultados.append({
                        "score": score,
                        "arquivo": item["arquivo"],
                        "planilha": item["planilha"],
                        "texto": item["texto"]
                    })

            resultados.sort(
                key=lambda x: x["score"],
                reverse=True
            )

            resultados = resultados[
                :quantidade_resultados
            ]

            tempo_pesquisa = (
                time.time() - inicio_pesquisa
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
                        f"— Similaridade "
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
                            f"**Similaridade:** "
                            f"{resultado['score']:.1f}%"
                        )

                        st.text_area(
                            "Conteúdo encontrado:",
                            resultado["texto"][:5000],
                            height=250,
                            key=f"resultado_{numero}"
                        )


# ============================================================
# PRÓXIMA ETAPA
# ============================================================

st.divider()

st.header("🚀 Próxima etapa do POC")

st.write(
    """
Depois de validarmos a pesquisa nas APRs reais, vamos evoluir o sistema para:

1. Encontrar automaticamente as APRs mais semelhantes;
2. Identificar perigos e riscos relacionados;
3. Identificar controles existentes;
4. Consultar NRs e requisitos aplicáveis;
5. Montar uma nova APR;
6. Preservar o layout padrão da empresa;
7. Gerar a APR em Excel;
8. Medir se conseguimos atingir a meta de **≤ 5 minutos por APR**.
"""
)

st.caption(
    "POC — A pesquisa atual utiliza a base fornecida pelo usuário. "
    "A geração automática por IA será implementada na próxima etapa."
)
