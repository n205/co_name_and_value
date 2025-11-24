import logging
import requests
import numpy as np
from io import BytesIO
from pypdf import PdfReader
from pdf2image import convert_from_bytes
import warnings
from gspread_dataframe import get_as_dataframe
import google.generativeai as genai
import os


# -------------------------------
# Gemini 初期化（共通）
# -------------------------------
def init_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("環境変数 GEMINI_API_KEY が設定されていません")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.0-flash")


text_model = None
image_model = None


# ============================================================
#  1) バリュー（テキスト版）抽出
# ============================================================
def extract_value_from_text(pdf_bytes):
    global text_model
    if text_model is None:
        text_model = init_gemini()

    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        all_text = ""

        for i in range(min(10, len(reader.pages))):
            text = reader.pages[i].extract_text()
            if text:
                all_text += text + "\n"

        if not all_text.strip():
            return "取得失敗"

        prompt = """
        以下は企業の統合報告書です。
        この中から企業が提示している「バリュー」「行動指針」「価値観」「行動規範」に該当する内容を150文字以内で要約してください。

        ・社員がどのような行動や姿勢を求められているかを優先
        ・説明文、前置き、ラベルは禁止
        ・内容そのものだけを返す
        ・取得できない場合は「取得失敗」
        """

        response = text_model.generate_content([prompt, all_text])
        result = response.text.strip()

        return result if result else "取得失敗"

    except Exception as e:
        warnings.warn(f"Geminiテキスト処理失敗: {e}")
        return "取得失敗"


# ============================================================
#  update_バリューT（テキスト）
# ============================================================
def update_バリューT(worksheet):
    logging.info("🧭 update_バリューT 開始")

    df = get_as_dataframe(worksheet)
    df.fillna('', inplace=True)

    if 'バリューT' not in df.columns:
        df['バリューT'] = ''

    update_count = 0

    for idx, row in df.iterrows():
        url = row.get("URL", "")
        val_t = row.get("バリューT", "")
        company = row.get("会社名", "")

        if not url or val_t:
            continue

        if company in ["対象外", "取得失敗", ""]:
            df.at[idx, "バリューT"] = "対象外"
            update_count += 1
            logging.info(f"⏭️ 対象外（会社名）: {url}")
            continue

        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            res = requests.get(url, headers=headers, timeout=20)

            if res.status_code == 200:
                extracted = extract_value_from_text(res.content)
                df.at[idx, "バリューT"] = extracted
                update_count += 1
                logging.info(f"📝 抽出(T): {url} → {extracted}")

            else:
                df.at[idx, "バリューT"] = "取得失敗"
                update_count += 1
                logging.warning(f"⚠️ DL失敗 {res.status_code}: {url}")

        except Exception as e:
            df.at[idx, "バリューT"] = "取得失敗"
            update_count += 1
            logging.warning(f"❌ 例外発生 {e}: {url}")

    df.replace([np.nan, np.inf, -np.inf], '', inplace=True)

    # Excel 列名変換
    def col_to_letter(index):
        letters = ""
        while index >= 0:
            index, rem = divmod(index, 26)
            letters = chr(65 + rem) + letters
            index -= 1
        return letters

    col_index = df.columns.get_loc("バリューT")
    col_letter = col_to_letter(col_index)

    worksheet.update(
        f"{col_letter}2:{col_letter}{len(df)+1}",
        [[v] for v in df["バリューT"].tolist()]
    )

    logging.info(f"📝 {update_count} 件のバリューTを更新")
    return f"{update_count} 件更新", 200


# ============================================================
#  2) バリュー（画像版）抽出
# ============================================================
def extract_value_from_pdf(pdf_bytes):
    global image_model
    if image_model is None:
        image_model = init_gemini()

    try:
        images = convert_from_bytes(pdf_bytes, dpi=200, first_page=1, last_page=10)

        prompt = """
        この画像は会社の統合報告書の最初の数ページです。
        会社が記載しているバリュー(Value)、価値観、行動指針、行動規範などの「中身」を150文字以内にまとめてください。

        ・社員に求められる姿勢・行動を優先
        ・説明は禁止
        ・ラベル（バリュー等）は不要
        ・取得できない場合は「取得失敗」
        """

        response = image_model.generate_content([prompt, *images])
        result = response.text.strip()

        return result if result else "取得失敗"

    except Exception as e:
        warnings.warn(f"Gemini画像処理失敗: {e}")
        return "取得失敗"


# ============================================================
#  update_バリューG（画像）
# ============================================================
def update_バリューG(worksheet):
    logging.info("🖼️ update_バリューG 開始")

    df = get_as_dataframe(worksheet)
    df.fillna('', inplace=True)

    if 'バリューG' not in df.columns:
        df['バリューG'] = ''

    update_count = 0

    for idx, row in df.iterrows():
        url = row.get("URL", "")
        val_g = row.get("バリューG", "")
        company = row.get("会社名", "")

        if not url or val_g:
            continue

        if company in ["対象外", "取得失敗", ""]:
            df.at[idx, "バリューG"] = "対象外"
            update_count += 1
            logging.info(f"⏭️ 対象外（会社名）: {url}")
            continue

        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            res = requests.get(url, headers=headers, timeout=20)

            if res.status_code == 200:
                extracted = extract_value_from_pdf(res.content)
                df.at[idx, "バリューG"] = extracted
                update_count += 1
                logging.info(f"🖼️ 抽出(G): {url} → {extracted}")

            else:
                df.at[idx, "バリューG"] = "取得失敗"
                update_count += 1
                logging.warning(f"⚠️ DL失敗 {res.status_code}: {url}")

        except Exception as e:
            df.at[idx, "バリューG"] = "取得失敗"
            update_count += 1
            logging.warning(f"❌ 例外発生 {e}: {url}")

    df.replace([np.nan, np.inf, -np.inf], '', inplace=True)

    # Excel列名計算
    def col_to_letter(index):
        letters = ""
        while index >= 0:
            index, rem = divmod(index, 26)
            letters = chr(65 + rem) + letters
            index -= 1
        return letters

    col_index = df.columns.get_loc("バリューG")
    col_letter = col_to_letter(col_index)

    worksheet.update(
        f"{col_letter}2:{col_letter}{len(df)+1}",
        [[v] for v in df["バリューG"].tolist()]
    )

    logging.info(f"📝 {update_count} 件のバリューGを更新")
    return f"{update_count} 件更新", 200


# ============================================================
#  バリュー統合処理
# ============================================================
def merge_values(value_t, value_g):
    """T/G を統合して1つのバリューを作成"""

    def is_valid(v):
        return v and v not in ["取得失敗", "対象外"]

    # 片方だけ有効 → それだけを採用
    if is_valid(value_t) and not is_valid(value_g):
        return value_t

    if is_valid(value_g) and not is_valid(value_t):
        return value_g

    # 両方無効
    if not is_valid(value_t) and not is_valid(value_g):
        return "取得失敗"

    # 両方有効 → Gemini で統合
    global merge_model
    if merge_model is None:
        merge_model = init_merge_model()

    prompt = f"""
    次の2つの抽出結果をもとに、企業の「バリュー」「価値観」「行動指針」に該当する内容を
    200文字以内で統合してください。

    ---
    抽出1: {value_t}
    抽出2: {value_g}
    ---

    条件:
    ・説明文やラベルは不要、内容だけを返す
    ・社員に求められる姿勢や行動が伝わる内容を優先
    ・70文字未満の場合は「取得失敗」と返す
    ・統合に失敗した場合も「取得失敗」と返す
    """

    try:
        res = merge_model.generate_content(prompt)
        result = res.text.strip()

        if not result or len(result) < 70:
            return "取得失敗"

        return result

    except Exception as e:
        logging.warning(f"❌ Geminiマージ失敗: {e}")
        return "取得失敗"


# ============================================================
#  update_バリュー
# ============================================================
def update_バリュー(worksheet):

    logging.info("🔄 update_バリュー 開始")

    df = get_as_dataframe(worksheet)
    df.fillna('', inplace=True)

    # 列がなければ追加
    if "バリュー" not in df.columns:
        df["バリュー"] = ""

    update_count = 0

    for idx, row in df.iterrows():
        current = row.get("バリュー", "")
        company = row.get("会社名", "")

        # すでに入力済みなら飛ばす
        if current:
            continue

        # 対象外企業
        if company == "対象外":
            df.at[idx, "バリュー"] = "対象外"
            update_count += 1
            logging.info(f"⏭️ 対象外（会社名）: {row.get('URL', '')}")
            continue

        merged = merge_values(row.get("バリューT", ""), row.get("バリューG", ""))
        df.at[idx, "バリュー"] = merged
        update_count += 1

        short = merged[:30] + ("…" if len(merged) > 30 else "")
        logging.info(f"📝 {row.get('URL', '')} → {short}")

    df.replace([np.nan, np.inf, -np.inf], "", inplace=True)

    # Excel列名変換（AA対応）
    def col_to_letter(index):
        letters = ''
        while index >= 0:
            index, rem = divmod(index, 26)
            letters = chr(65 + rem) + letters
            index -= 1
        return letters

    col_index = df.columns.get_loc("バリュー")
    col_letter = col_to_letter(col_index)

    worksheet.update(
        f"{col_letter}2:{col_letter}{len(df)+1}",
        [[v] for v in df["バリュー"].tolist()]
    )

    logging.info(f"📝 update_バリュー: {update_count} 件 更新完了")

    return f"{update_count} 件更新", 200
