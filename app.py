import streamlit as st
import tempfile
import os
import birdnet

# -----------------------------
# ページ設定（アイコン変更）
# -----------------------------
st.set_page_config(
    page_title="ピヨピヨ判定くん",
    page_icon="🐦",
    layout="centered"
)

st.title("🐦 ピヨピヨ判定くん｜鳥の鳴き声解析アプリ")

# -----------------------------
# 日本語変換辞書
# -----------------------------
bird_translation = {
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
    "Warbling White-eye": "メジロ",
}

def translate_bird(name):
    for eng, jp in bird_translation.items():
        if eng in name:
            return jp
    return name

# -----------------------------
# モデル読み込み
# -----------------------------
@st.cache_resource
def load_model():
    return birdnet.load("acoustic", "2.4", "tf")

model = load_model()

# -----------------------------
# 入力方法（マイクをデフォルト）
# -----------------------------
input_mode = st.radio(
    "音声入力方法を選択してください",
    ["マイクで録音", "ファイルをアップロード"],
    index=0
)

audio_bytes = None

if input_mode == "ファイルをアップロード":
    uploaded = st.file_uploader("音声ファイルを選択", type=["wav", "mp3"])
    if uploaded:
        audio_bytes = uploaded.read()

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

            name_en = top["species_name"]
            name = translate_bird(name_en)

            confidence_percent = top["confidence"] * 100

            st.success(
                f"{confidence_percent:.1f}%の確率で{name}です"
            )

        else:
            st.warning("鳥を検出できませんでした。")

    except Exception as e:
        st.error("エラーが発生しました")
        st.write(e)

    finally:
        os.remove(tmp_path)
