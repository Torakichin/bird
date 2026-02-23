import streamlit as st
import tempfile
import os
import pandas as pd
import birdnet

st.title("🐦 BirdNET 動作確認アプリ")

# -----------------------------
# モデルロード（旧birdnet固定）
# -----------------------------
@st.cache_resource
def load_model():
    return birdnet.load("acoustic", "2.4", "tf")

model = load_model()

# -----------------------------
# 音声アップロードのみ（まずは録音なし）
# -----------------------------
uploaded = st.file_uploader("音声ファイルをアップロード", type=["wav", "mp3"])

if uploaded:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    st.info("解析中...")

    try:
        predictions = model.predict(tmp_path)

        # デバッグ用：型確認
        st.write("返り値の型:", type(predictions))

        # DataFrame想定（birdnet 0.2.11）
        if hasattr(predictions, "empty"):

            if not predictions.empty:
                st.write("予測結果（上位5件）")
                st.dataframe(
                    predictions.sort_values("confidence", ascending=False).head()
                )

                top = predictions.sort_values(
                    "confidence", ascending=False
                ).iloc[0]

                english_name = top["common_name"]
                confidence = top["confidence"]

                st.success(f"Top Prediction: {english_name}")
                st.write(f"Confidence: {confidence:.3f}")

            else:
                st.warning("予測結果が空です")

        else:
            st.warning("予測結果がDataFrameではありません")
            st.write(predictions)

    except Exception as e:
        st.error("エラーが発生しました")
        st.write(e)

    finally:
        os.remove(tmp_path)
