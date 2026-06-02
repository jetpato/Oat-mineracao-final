"""
==================================================================
PROJETO SUPERMERCADO — Sistema de Recomendação de Produtos
==================================================================
Grupo: Alisson Silva, Ana Luiza Lima, Julio Cesar Brito,
       João Pedro Ceo, Mateus Dantas, Levi Ramos
Disciplina: Mineração de Dados | Profa. Bruna Silva Celestino
Curso: Sistemas de Informação — UNEX

Modelo: Algoritmo Apriori (Regras de Associação)

Como executar:
    pip install streamlit mlxtend joblib pandas
    streamlit run app.py
==================================================================
"""

import streamlit as st
import joblib
import pandas as pd
import os

st.set_page_config(
    page_title="Recomendador de Produtos — Supermercado",
    page_icon="🛒",
    layout="wide",
)

INTEGRANTES = [
    "Alisson Silva", "Ana Luiza Lima", "Julio Cesar Brito",
    "João Pedro Ceo", "Mateus Dantas", "Levi Ramos",
]

st.markdown(
    """
    <style>
    .titulo-grupo { font-size: 0.9rem; color: #9aa0a6; margin-top: -10px; }
    .card-rec {
        background: linear-gradient(135deg, #1e3a2f 0%, #14532d 100%);
        border-radius: 14px; padding: 24px; text-align: center;
        border: 1px solid #2e7d32;
    }
    .card-rec h2 { color: #a7f3d0; margin: 0; font-size: 1.1rem; }
    .card-rec .prod { font-size: 2.2rem; font-weight: 700; color: #ffffff; }
    .emoji-prod { font-size: 3rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def carregar_modelo():
    if not os.path.exists("modelo_supermercado.pkl"):
        return None
    return joblib.load("modelo_supermercado.pkl")


modelo = carregar_modelo()

EMOJIS = {
    "Arroz": "🍚", "Feijão": "🫘", "Óleo": "🛢️", "Pão": "🍞",
    "Leite": "🥛", "Café": "☕", "Carne": "🥩", "Manteiga": "🧈",
}

def emoji(produto):
    return EMOJIS.get(produto, "🛒")


# ------------------------------------------------------------------
# RECOMENDAÇÃO POR REGRAS DE ASSOCIAÇÃO
# ------------------------------------------------------------------
def recomendar_por_regras(carrinho, regras_df, top_n=5):
    entrada = frozenset(p.strip().capitalize() for p in carrinho)
    recs = {}
    for _, r in regras_df.iterrows():
        if r["antecedents"].issubset(entrada):
            for prod in r["consequents"] - entrada:
                if prod not in recs or r["confidence"] > recs[prod]["confianca"]:
                    recs[prod] = {
                        "produto": prod,
                        "confianca": float(r["confidence"]),
                        "lift": float(r["lift"]),
                        "origem": "regra",
                    }
    return sorted(recs.values(), key=lambda x: x["confianca"], reverse=True)[:top_n]


# ------------------------------------------------------------------
# RECOMENDAÇÃO DE RESERVA — por popularidade
# (usada quando nenhuma regra de associação é acionada)
# ------------------------------------------------------------------
def recomendar_por_popularidade(carrinho, suporte_produtos, top_n=3):
    entrada = set(p.strip().capitalize() for p in carrinho)
    candidatos = [
        {"produto": prod, "confianca": sup, "lift": None, "origem": "popularidade"}
        for prod, sup in suporte_produtos.items()
        if prod not in entrada
    ]
    return sorted(candidatos, key=lambda x: x["confianca"], reverse=True)[:top_n]


# ==================================================================
# SIDEBAR
# ==================================================================
with st.sidebar:
    st.header("📚 Sobre o Projeto")
    st.write(
        "Sistema de recomendação de produtos para supermercado, "
        "usando o algoritmo **Apriori** de Regras de Associação."
    )
    st.divider()
    st.subheader("👥 Integrantes do Grupo")
    for nome in INTEGRANTES:
        st.markdown(f"- {nome}")
    st.divider()
    if modelo is not None:
        st.subheader("⚙️ Parâmetros do Modelo")
        st.metric("Suporte mínimo", f"{modelo['min_support']:.0%}")
        st.metric("Confiança mínima", f"{modelo['min_confidence']:.0%}")
        st.metric("Regras geradas", len(modelo["regras"]))

# ==================================================================
# CABEÇALHO
# ==================================================================
st.title("🛒 Recomendador de Produtos — Supermercado")
st.markdown(
    f"<p class='titulo-grupo'>Disciplina: Mineração de Dados — Profa. Bruna Silva Celestino | "
    f"Grupo: {', '.join(INTEGRANTES)}</p>",
    unsafe_allow_html=True,
)
st.markdown(
    "Selecione os produtos que o cliente já colocou no carrinho. "
    "O sistema vai recomendar o que ele tende a comprar também."
)
st.divider()

if modelo is None:
    st.error(
        "⚠️ Arquivo **modelo_supermercado.pkl** não encontrado.\n\n"
        "Coloque o arquivo na mesma pasta deste app.py e recarregue a página."
    )
    st.stop()

regras = modelo["regras"]
produtos = modelo["produtos"]
suporte_produtos = modelo.get("suporte_produtos", {})

# ==================================================================
# ENTRADA + RESULTADO
# ==================================================================
col_entrada, col_resultado = st.columns([1, 1.3], gap="large")

with col_entrada:
    st.subheader("🧺 Carrinho do cliente")
    carrinho = st.multiselect(
        "Selecione os produtos:",
        options=produtos,
        default=["Arroz"] if "Arroz" in produtos else None,
    )
    if carrinho:
        st.markdown("**No carrinho:**")
        st.markdown(" ".join(f"{emoji(p)} {p}" for p in carrinho))
    botao = st.button("🔍 Gerar recomendação", type="primary", use_container_width=True)

with col_resultado:
    st.subheader("📊 Resultado")

    if not botao:
        st.info("👈 Monte o carrinho e clique em **Gerar recomendação**.")
    elif not carrinho:
        st.warning("Selecione pelo menos um produto.")
    else:
        # 1) Tenta recomendar pelas regras de associação
        recomendacoes = recomendar_por_regras(carrinho, regras)
        usou_reserva = False

        # 2) Caso especial: carrinho já tem todos os produtos
        produtos_fora = [p for p in produtos if p not in carrinho]
        if not produtos_fora:
            st.info(
                "🛒 O carrinho já contém **todos os produtos disponíveis**! "
                "Não há mais o que recomendar — o cliente levou tudo."
            )
            st.stop()

        # 3) Se nenhuma regra foi acionada, usa a recomendação de reserva (popularidade)
        if not recomendacoes:
            recomendacoes = recomendar_por_popularidade(carrinho, suporte_produtos)
            usou_reserva = True

        if not recomendacoes:
            st.info("Não há recomendações disponíveis para esta combinação.")
            st.stop()

        melhor = recomendacoes[0]

        # ---- Cartão de destaque ----
        st.markdown(
            f"""
            <div class="card-rec">
                <div class="emoji-prod">{emoji(melhor['produto'])}</div>
                <h2>Recomendamos também:</h2>
                <div class="prod">{melhor['produto']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")

        if not usou_reserva:
            # Recomendação baseada em regra de associação
            c1, c2 = st.columns(2)
            c1.metric("Confiança", f"{melhor['confianca']:.0%}",
                      help="De cada 100 clientes que levaram esses produtos, "
                           f"cerca de {melhor['confianca']*100:.0f} também levaram a recomendação.")
            c2.metric("Lift", f"{melhor['lift']:.2f}x",
                      help="Quantas vezes mais provável é a compra conjunta vs. o acaso.")
            st.progress(melhor["confianca"],
                        text=f"Nível de confiança: {melhor['confianca']:.0%}")
            st.success(
                f"✅ Quem comprou **{', '.join(carrinho)}** também costuma "
                f"levar **{melhor['produto']}**!"
            )
        else:
            # Recomendação de reserva por popularidade
            st.metric("Popularidade", f"{melhor['confianca']:.0%}",
                      help="Proporção das compras que incluem este produto.")
            st.progress(melhor["confianca"],
                        text=f"Comprado em {melhor['confianca']:.0%} das transações")
            st.info(
                f"ℹ️ Para **{', '.join(carrinho)}** não há uma regra de associação forte. "
                f"Como sugestão geral, **{melhor['produto']}** é um dos produtos mais "
                "comprados na loja."
            )

        # ---- Tabela com todas as recomendações ----
        if len(recomendacoes) > 1:
            st.markdown("##### Outras sugestões:")
            linhas = []
            for r in recomendacoes:
                if r["origem"] == "regra":
                    forca = ("🟢 Muito forte" if r["lift"] >= 2 else
                             "🟡 Moderada" if r["lift"] >= 1.3 else "🔴 Fraca")
                    linhas.append({
                        "Produto": f"{emoji(r['produto'])} {r['produto']}",
                        "Confiança": f"{r['confianca']:.0%}",
                        "Lift": f"{r['lift']:.2f}x",
                        "Força": forca,
                    })
                else:
                    linhas.append({
                        "Produto": f"{emoji(r['produto'])} {r['produto']}",
                        "Confiança": f"{r['confianca']:.0%}",
                        "Lift": "—",
                        "Força": "⭐ Popular",
                    })
            st.dataframe(pd.DataFrame(linhas), hide_index=True, use_container_width=True)

st.divider()
st.caption(
    f"Modelo: Algoritmo Apriori (Regras de Associação)  •  "
    f"Suporte mínimo {modelo['min_support']:.0%}  •  "
    f"Confiança mínima {modelo['min_confidence']:.0%}  •  "
    f"Grupo: {', '.join(INTEGRANTES)}"
)
