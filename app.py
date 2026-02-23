import streamlit as st
import tempfile
import os
import birdnet

st.title("🐦 BirdNET 鳥類音声解析")

# -----------------------------
# モデル読み込み
# -----------------------------
@st.cache_resource
def load_model():
    return birdnet.load("acoustic", "2.4", "tf")

model = load_model()

# -----------------------------
# 入力方法の選択
# -----------------------------
input_mode = st.radio(
    "音声入力方法を選択してください",
    ["ファイルをアップロード", "マイクで録音"]
)

audio_bytes = None

# -----------------------------
# ① ファイルアップロード
# -----------------------------
if input_mode == "ファイルをアップロード":
    uploaded = st.file_uploader("音声ファイルを選択", type=["wav", "mp3"])
    if uploaded:
        audio_bytes = uploaded.read()

# -----------------------------
# ② マイク録音
# -----------------------------
if input_mode == "マイクで録音":
    recorded = st.audio_input("録音ボタンを押して鳥の声を録音してください")
    if recorded:
        audio_bytes = recorded.read()

# -----------------------------
# 解析処理
# -----------------------------
if audio_bytes:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    st.audio(audio_bytes)
    st.info("解析中...")

    try:
        predictions = model.predict(tmp_path)
        df = predictions.to_dataframe()

        if not df.empty:
            df_sorted = df.sort_values("confidence", ascending=False)
            top = df_sorted.iloc[0]

            name = top["species_name"]
            confidence = top["confidence"]

            st.success(f"Top Prediction: {name}")
            st.write(f"Confidence: {confidence:.3f}")

        else:
            st.warning("鳥を検出できませんでした。")

    except Exception as e:
        st.error("エラーが発生しました")
        st.write(e)

    finally:
        os.remove(tmp_path)
