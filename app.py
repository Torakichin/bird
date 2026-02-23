import streamlit as st
import tempfile
import os
import pandas as pd
import birdnet

st.title("🐦 BirdNET 動作確認アプリ")

@st.cache_resource
def load_model():
    return birdnet.load("acoustic", "2.4", "tf")

model = load_model()

uploaded = st.file_uploader("音声ファイルをアップロード", type=["wav", "mp3"])

if uploaded:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    st.info("解析中...")

    try:
        predictions = model.predict(tmp_path)
        df = predictions.to_dataframe()

        st.write("列名確認:", df.columns)

        if not df.empty:

            df_sorted = df.sort_values("confidence", ascending=False)
            top = df_sorted.iloc[0]

            # 🔥 列名吸収ロジック
            if "common_name" in df.columns:
                name = top["common_name"]
            elif "scientific_name" in df.columns:
                name = top["scientific_name"]
            elif "species" in df.columns:
                name = top["species"]
            elif "label" in df.columns:
                name = top["label"]
            else:
                name = "UNKNOWN_COLUMN"

            confidence = top["confidence"]

            st.success(f"Top Prediction: {name}")
            st.write(f"Confidence: {confidence:.3f}")

        else:
            st.warning("予測結果が空です")

    except Exception as e:
        st.error("エラーが発生しました")
        st.write(e)

    finally:
        os.remove(tmp_path)
