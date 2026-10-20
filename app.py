import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="HUB Lighting Finder",
    page_icon="💡",
    layout="wide"
)

st.title("💡 HUB Lighting Finder")

uploaded = st.file_uploader(
    "Ανέβασε αρχείο Excel προϊόντων",
    type=["xlsx", "csv"]
)

if uploaded:

    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

    st.sidebar.header("Φίλτρα")

    if "Brand" in df.columns:
        brands = st.sidebar.multiselect(
            "Εταιρεία",
            sorted(df["Brand"].dropna().unique())
        )

        if brands:
            df = df[df["Brand"].isin(brands)]

    if "Type" in df.columns:
        types = st.sidebar.multiselect(
            "Τύπος",
            sorted(df["Type"].dropna().unique())
        )

        if types:
            df = df[df["Type"].isin(types)]

    keyword = st.sidebar.text_input(
        "Αναζήτηση"
    )

    if keyword:
        keyword = keyword.lower()

        mask = df.astype(str).apply(
            lambda x: x.str.lower().str.contains(keyword)
        ).any(axis=1)

        df = df[mask]

    st.metric(
        "Αποτελέσματα",
        len(df)
    )

    st.dataframe(
        df,
        use_container_width=True
    )

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Export CSV",
        csv,
        "results.csv",
        "text/csv"
    )

else:
    st.info(
        "Ανέβασε Excel με προϊόντα."
    )
