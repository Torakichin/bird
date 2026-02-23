import streamlit as st
import tempfile
import os
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

        st.write("列一覧:", list(df.columns))

        if not df.empty:

            df_sorted = df.sort_values("confidence", ascending=False)
            top = df_sorted.iloc[0]

            # -----------------------------
            # 種名抽出ロジック（完全版）
            # -----------------------------
            name = None

            # ① 列にある場合
            for col in ["common_name", "scientific_name", "species", "label"]:
                if col in df.columns:
                    name = top[col]
                    break

            # ② indexに入っている場合
            if name is None:
                name = top.name  # ← ここが重要

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
