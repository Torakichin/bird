import streamlit as st
import tempfile
import os
import pandas as pd

# 旧birdnet優先
try:
    import birdnet
    USE_LEGACY = True
except ImportError:
    USE_LEGACY = False

# 新API fallback
if not USE_LEGACY:
    try:
        from birdnet_analyzer import Analyzer
        USE_ANALYZER = True
    except ImportError:
        USE_ANALYZER = False
else:
    USE_ANALYZER = False


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
# モデルロード
# -----------------------------
@st.cache_resource
def load_model():
    if USE_LEGACY:
        return birdnet.load("acoustic", "2.4", "tf")
    elif USE_ANALYZER:
        return Analyzer()
    else:
        st.error("BirdNETがインストールされていません")
        st.stop()

model = load_model()

uploaded = st.file_uploader("音声ファイルをアップロード", type=["wav", "mp3"])

if uploaded:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    st.info("解析中...")

    # -----------------------------
    # 推論
    # -----------------------------
    if USE_LEGACY:
        predictions = model.predict(
            tmp_path,
            custom_species_list="species_list.txt",
        )
    else:
        predictions = model.analyze(tmp_path)

    english_name = None
    confidence = None

    # -----------------------------
    # DataFrame型
    # -----------------------------
    if hasattr(predictions, "empty"):
        if not predictions.empty:
            top = predictions.sort_values("confidence", ascending=False).iloc[0]
            english_name = top["common_name"]
            confidence = top["confidence"]

    # -----------------------------
    # list型
    # -----------------------------
    elif isinstance(predictions, list):
        if len(predictions) > 0:
            top = sorted(predictions, key=lambda x: x["confidence"], reverse=True)[0]
            english_name = top["common_name"]
            confidence = top["confidence"]

    # -----------------------------
    # dict型
    # -----------------------------
    elif isinstance(predictions, dict):
        if "predictions" in predictions and len(predictions["predictions"]) > 0:
            top = sorted(
                predictions["predictions"],
                key=lambda x: x["confidence"],
                reverse=True,
            )[0]
            english_name = top["common_name"]
            confidence = top["confidence"]

    # -----------------------------
    # 出力
    # -----------------------------
    if english_name:
        jp_name = JP_NAME.get(english_name, english_name)
        st.success(f"🐦 推定種: {jp_name}")
        st.write(f"信頼度: {confidence:.2f}")
    else:
        st.warning("鳥を検出できませんでした。")

    os.remove(tmp_path)
