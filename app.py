import streamlit as st
import tempfile
import os
import pandas as pd
import birdnet

st.title("🐦 BirdNET 動作確認アプリ")

# -----------------------------
# モデルロード
# -----------------------------
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
        # 予測実行
        predictions = model.predict(tmp_path)

        # 🔥 ここが重要
        df = predictions.to_dataframe()

        if not df.empty:

            df_sorted = df.sort_values("confidence", ascending=False)

            st.write("上位5件")
            st.dataframe(df_sorted.head())

            top = df_sorted.iloc[0]
            english_name = top["common_name"]
            confidence = top["confidence"]

            st.success(f"Top Prediction: {english_name}")
            st.write(f"Confidence: {confidence:.3f}")

        else:
            st.warning("予測結果が空です")

    except Exception as e:
        st.error("エラーが発生しました")
        st.write(e)

    finally:
        os.remove(tmp_path)
