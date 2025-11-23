import logging
import requests
import numpy as np
from datetime import datetime
from pypdf import PdfReader
from io import BytesIO
import warnings
from gspread_dataframe import get_as_dataframe

import google.generativeai as genai
import os


# --- Gemini 初期化 ---
def init_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("環境変数 GEMINI_API_KEY が設定されていません")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.0-flash")


text_model = None


def extract_company_name_from_text(pdf_bytes):
    """PDF テキストから会社名を Gemini で抽出する"""
    global text_model

    try:
        if text_model is None:
            text_model = init_gemini()

        reader = PdfReader(BytesIO(pdf_bytes))
        all_text = ''

        # 最初の 3 ページのみ使用
        for i in range(min(3, len(reader.pages))):
            page = reader.pages[i]
            page_text = page.extract_text()
            if page_text:
                all_text += page_text + '\n'

        if not all_text.strip():
            return "取得失敗"

        prompt = """
        以下は統合報告書の最初の数ページです。
        この中から「会社名」を 1 行で抽出してください。

        - 「株式会社○○」「○○株式会社」形式が多い
        - 出力には法人格（株式会社等）を含めない
        - 補足、記号、説明は不要
        - 取得に失敗した場合は「取得失敗」と返す
        """

        response = text_model.generate_content([prompt, all_text])
        result = response.text.strip()

        return result if result else "取得失敗"

    except Exception as e:
        logging.warning(f"Geminiテキスト処理失敗（会社名T）: {e}")
        return "取得失敗"


# --- メイン処理 ---
def update_company_name_t(worksheet):
    logging.info("🏢 update_company_name_t 開始")

    df = get_as_dataframe(worksheet)
    df.fillna('', inplace=True)
    update_count = 0

    for idx, row in df.iterrows():
        url = row['URL']
        name_t = row.get('会社名T', '')
        page_count = row['ページ数']

        # URL なし or すでに記入済みはスキップ
        if not url or name_t:
            continue

        # ページ数 15 以下は対象外
        if isinstance(page_count, (int, float)) and page_count <= 15:
            df.at[idx, '会社名T'] = '対象外'
            update_count += 1
            logging.info(f"⏭️ ページ数少 → 対象外: {url}")
            continue

        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=15)

            if response.status_code == 200:
                extracted = extract_company_name_from_text(response.content)
                df.at[idx, '会社名T'] = extracted
                logging.info(f"🔍 会社名T: {url} → {extracted}")
            else:
                df.at[idx, '会社名T'] = '取得失敗'
                logging.info(f"⚠️ PDF取得失敗 → 取得失敗: {url}")

        except Exception as e:
            df.at[idx, '会社名T'] = '取得失敗'
            logging.warning(f"❌ エラー → 取得失敗: {e} → {url}")

        update_count += 1

    # NaN クリーニング
    df.replace([np.nan, np.inf, -np.inf], '', inplace=True)

    # 列位置 → Excel 形式へ
    col_index = df.columns.get_loc('会社名T')
    col_letter = chr(ord('A') + col_index)

    # シート更新
    if update_count > 0:
        worksheet.update(
            f"{col_letter}2:{col_letter}{len(df)+1}",
            [[value] for value in df['会社名T'].tolist()]
        )
        logging.info(f"📄 {update_count} 件の会社名Tを更新")
    else:
        logging.info("🔁 更新なし")

    return f"{update_count} 件の会社名T更新", 200
