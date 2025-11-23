import logging
import requests
import numpy as np
from io import BytesIO
from pypdf import PdfReader
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


# ============================================================
#  1) バリューのテキスト抽出
# ============================================================
def extract_value_from_text(pdf_bytes):
    global text_model
    if text_model is None:
        text_model = init_gemini()

    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        all_text = ''

        # 10ページ以内を対象
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
#  update_バリューT（Cloud Functions 用）
# ============================================================
def update_バリューT(worksheet):
    logging.info("🧭 update_バリューT 開始")

    df = get_as_dataframe(worksheet)
    df.fillna('', inplace=True)

    # バリューT列が無ければ作成
    if 'バリューT' not in df.columns:
        df['バリューT'] = ''

    update_count = 0

    for idx, row in df.iterrows():
        url = row.get("URL", "")
        val_t = row.get("バリューT", "")
        company = row.get("会社名", "")

        # URLなし or 既に抽出済 → スキップ
        if not url or val_t:
            continue

        # 会社名が対象外ならバリューも対象外
        if company in ["対象外", "取得失敗", ""]:
            df.at[idx, "バリューT"] = "対象外"
            update_count += 1
            logging.info(f"⏭️ 対象外（会社名）: {url}")
            continue

        # PDF ダウンロード
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            res = requests.get(url, headers=headers, timeout=20)

            if res.status_code == 200:
                extracted = extract_value_from_text(res.content)
                df.at[idx, "バリューT"] = extracted
                update_count += 1

                if extracted == "取得失敗":
                    logging.info(f"⚠️ 抽出失敗: {url}")
                else:
                    logging.info(f"✅ 抽出成功: {url}")

            else:
                df.at[idx, "バリューT"] = "取得失敗"
                update_count += 1
                logging.warning(f"⚠️ DL失敗 {res.status_code}: {url}")

        except Exception as e:
            df.at[idx, "バリューT"] = "取得失敗"
            update_count += 1
            logging.warning(f"❌ 例外発生 {e}: {url}")

    # NaN→空白
    df.replace([np.nan, np.inf, -np.inf], '', inplace=True)

    # 列番号 → Excel 形式
    def column_index_to_letter(index):
        letters = ""
        while index >= 0:
            index, remainder = divmod(index, 26)
            letters = chr(65 + remainder) + letters
            index -= 1
        return letters

    col_index = df.columns.get_loc("バリューT")
    col_letter = column_index_to_letter(col_index)

    # シート更新
    worksheet.update(
        f"{col_letter}2:{col_letter}{len(df)+1}",
        [[v] for v in df["バリューT"].tolist()]
    )

    logging.info(f"📝 {update_count} 件のバリューTを更新しました")

    return f"{update_count} 件更新", 200
