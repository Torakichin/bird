import streamlit as st
import birdnet
import tempfile
import os
import pandas as pd
from streamlit_mic_recorder import mic_recorder

st.title("🐦 鳥の音声識別アプリ（BirdNET）")

# -----------------------------
# 日本語名辞書
# -----------------------------
JP_NAME = {
    "Long-tailed Tit": "エナガ",
    "Northern Pintail": "オナガガモ",
    "Green-winged Teal": "コガモ",
    "Mallard": "マガモ",
    "Eastern Spot-billed Duck": "カルガモ",
    "Gray Heron": "アオサギ",
    "Common Pochard": "ホシハジロ",
    "Oriental Greenfinch": "カワラヒワ",
    "Large-billed Crow": "ハシブトガラス",
    "Little Egret": "コサギ",
    "Meadow Bunting": "ホオジロ",
    "Black-faced Bunting": "アオジ",
    "Eurasian Coot": "オオバン",
    "Japanese Bush Warbler": "ウグイス",
    "Brown-eared Bulbul": "ヒヨドリ",
    "Bull-headed Shrike": "モズ",
    "Eurasian Wigeon": "ヒドリガモ",
    "Black Kite": "トビ",
    "White Wagtail": "ハクセキレイ",
    "Japanese Wagtail": "セグロセキレイ",
    "Osprey": "ミサゴ",
    "Japanese Tit": "シジュウカラ",
    "Eurasian Tree Sparrow": "スズメ",
    "Great Cormorant": "カワウ",
    "Daurian Redstart": "ジョウビタキ",
    "Varied Tit": "ヤマガラ",
    "White-cheeked Starling": "ムクドリ",
    "Oriental Turtle-Dove": "キジバト",
    "Little Grebe": "カイツブリ",
    "Dusky Thrush": "ツグミ",
    "Pale Thrush": "シロハラ",
    "Japanese Pygmy Woodpecker": "コゲラ",
    "Warbling White-eye": "メジロ"
}

# -----------------------------
# モデルロード（初回のみ）
# -----------------------------
@st.cache_resource
def load_model():
    return birdnet.load("acoustic", "2.4", "tf")

model = load_model()

# -----------------------------
# 音声入力方法選択
# -----------------------------
option = st.radio("音声入力方法を選択してください", ["🎤 録音する", "📁 ファイルアップロード"])

audio_file = None

if option == "🎤 録音する":
    audio = mic_recorder(start_prompt="録音開始", stop_prompt="録音停止")
    if audio:
        audio_file = audio["bytes"]

else:
    uploaded = st.file_uploader("音声ファイルをアップロード", type=["wav", "mp3"])
    if uploaded:
        audio_file = uploaded.read()

# -----------------------------
# 推論
# -----------------------------
if audio_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file)
        tmp_path = tmp.name

    st.info("解析中...")

    predictions = model.predict(
        tmp_path,
        custom_species_list="species_list.txt"
    )

    df = predictions

    if not df.empty:
        top = df.sort_values("confidence", ascending=False).iloc[0]
        english_name = top["common_name"]
        confidence = top["confidence"]

        jp_name = JP_NAME.get(english_name, english_name)

        st.success(f"🐦 推定種: {jp_name}")
        st.write(f"信頼度: {confidence:.2f}")

    else:
        st.warning("鳥を検出できませんでした。")

    os.remove(tmp_path)
