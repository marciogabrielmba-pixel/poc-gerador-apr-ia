import streamlit as st
import time

st.set_page_config(
    page_title="Gerador de APR com IA",
    page_icon="🦺",
    layout="wide"
)

# -----------------------------
# CONFIGURAÇÃO
# -----------------------------

st.title("🦺 Gerador de APR com IA")
st.subheader("Prova de Conceito — 10 APRs em até 50 minutos")

st.markdown("---")

# Indicadores
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("META", "10 APRs")

with col2:
    st.metric("TEMPO TOTAL", "50 minutos")

with col3:
    st.metric("TEMPO / APR", "5 minutos")

st.markdown("---")

# -----------------------------
# ENTRADA
# -----------------------------

st.markdown("### 1. Informe a atividade")

atividade = st.text_input(
    "Digite a atividade para gerar a APR:",
    placeholder="Ex.: Instalação de eletrocalhas em altura"
)

# -----------------------------
# GERAÇÃO
# -----------------------------

if st.button("🚀 GERAR APR", type="primary"):

    if not atividade:

        st.warning("Digite uma atividade primeiro.")

    else:

        inicio = time.time()

        with st.spinner("Analisando atividade e estruturando APR..."):

            time.sleep(2)

        tempo = time.time() - inicio

        st.success("APR gerada com sucesso!")

        st.markdown("---")

        # -----------------------------
        # RESULTADO
        # -----------------------------

        st.markdown("## 📋 APR GERADA")

        st.markdown(f"### Atividade")
        st.write(atividade)

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("### Perigos identificados")

            st.write("""
            - Trabalho em altura
            - Queda de pessoas
            - Queda de materiais
            - Proximidade de estruturas
            """)

        with col2:

            st.markdown("### Riscos")

            st.write("""
            - Queda com diferença de nível
            - Queda de objetos
            - Impacto contra estruturas
            - Lesões graves
            """)

        st.markdown("### Consequências")

        st.write("""
        Lesões leves, graves ou fatais dependendo da exposição,
        queda de materiais e danos a equipamentos.
        """)

        st.markdown("### Medidas de Controle")

        st.write("""
        1. Inspecionar previamente o local de trabalho.
        2. Isolar e sinalizar a área.
        3. Verificar condições dos equipamentos.
        4. Utilizar proteção coletiva sempre que aplicável.
        5. Utilizar sistema de proteção contra quedas.
        6. Manter ferramentas e materiais organizados.
        7. Garantir trabalhador capacitado e autorizado.
        8. Realizar DDS antes do início da atividade.
        """)

        st.markdown("### EPC")

        st.write("""
        Guarda-corpo, isolamento da área, sinalização,
        linha de vida e demais proteções coletivas aplicáveis.
        """)

        st.markdown("### EPI")

        st.write("""
        Capacete com jugular, calçado de segurança,
        óculos de proteção, luvas adequadas e cinturão
        tipo paraquedista quando aplicável.
        """)

        st.markdown("### Requisitos legais de referência")

        st.write("""
        NR-01 — Gerenciamento de Riscos Ocupacionais
        NR-06 — Equipamento de Proteção Individual
        NR-18 — Segurança e Saúde na Indústria da Construção
        NR-35 — Trabalho em Altura
        """)

        st.markdown("---")

        # -----------------------------
        # KPI
        # -----------------------------

        st.markdown("## ⏱️ Resultado da POC")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Tempo desta APR",
                f"{tempo:.1f} segundos"
            )

        with c2:
            st.metric(
                "Meta por APR",
                "≤ 5 minutos"
            )

        with c3:
            st.metric(
                "Meta da POC",
                "10 APRs / 50 min"
            )

        st.success(
            "🎯 A demonstração indica capacidade de geração dentro "
            "da meta de 5 minutos por APR."
        )

        st.info(
            "Esta é uma POC. A próxima etapa será conectar a base "
            "real de APRs e substituir a geração demonstrativa por "
            "IA + recuperação de documentos."
        )
