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
# Gemini 初期化（1回のみ）
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
#  1) テキストで抽出（組織名T）
# ============================================================
def extract_company_name_from_text(pdf_bytes):
    global text_model
    if text_model is None:
        text_model = init_gemini()

    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        all_text = ''

        for i in range(min(3, len(reader.pages))):
            text = reader.pages[i].extract_text()
            if text:
                all_text += text + '\n'

        if not all_text.strip():
            return "取得失敗"

        prompt = """
        以下は統合報告書の最初の数ページです。
        この中から「会社名」を抽出してください。

        - 「株式会社〇〇」「〇〇株式会社」形式が多い
        - 出力には法人格を含めないこと
        - 補足説明・記号・文章は禁止
        - 1行のみで出す
        - 取得に失敗した場合は「取得失敗」
        """

        response = text_model.generate_content([prompt, all_text])
        result = response.text.strip()
        return result if result else "取得失敗"

    except Exception as e:
        logging.warning(f"Geminiテキスト処理失敗: {e}")
        return "取得失敗"


def update_組織名T(worksheet):
    logging.info("🏢 update_組織名T開始")

    df = get_as_dataframe(worksheet)
    df.fillna('', inplace=True)

    update_count = 0

    for idx, row in df.iterrows():
        url = row['URL']
        name_t = row.get('会社名T', '')
        page_count = row['ページ数']

        if not url or name_t:
            continue

        # ページ数制限
        if isinstance(page_count, (int, float)) and page_count <= 15:
            df.at[idx, '会社名T'] = '対象外'
            update_count += 1
            logging.info(f"⏭️ 対象外: {url}")
            continue

        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            res = requests.get(url, headers=headers, timeout=15)

            if res.status_code == 200:
                extracted = extract_company_name_from_text(res.content)
                df.at[idx, '会社名T'] = extracted
                logging.info(f"🔍 T抽出: {url} → {extracted}")
            else:
                df.at[idx, '会社名T'] = '取得失敗'
                logging.warning(f"⚠️ DL失敗 {url}")
        except Exception as e:
            df.at[idx, '会社名T'] = '取得失敗'
            logging.warning(f"❌ error: {e} {url}")

        update_count += 1

    # シート更新
    df.replace([np.nan, np.inf, -np.inf], '', inplace=True)
    col_index = df.columns.get_loc('会社名T')
    col_letter = chr(ord('A') + col_index)

    worksheet.update(
        f"{col_letter}2:{col_letter}{len(df)+1}",
        [[v] for v in df['会社名T'].tolist()]
    )
    logging.info(f"📄 {update_count} 件の会社名T更新")

    return f"{update_count} 件更新", 200


# ============================================================
#  2) 画像で抽出（組織名G）
# ============================================================
def extract_company_name_from_pdf_image(pdf_bytes):
    global image_model
    if image_model is None:
        image_model = init_gemini()

    try:
        images = convert_from_bytes(pdf_bytes, dpi=200, first_page=1, last_page=3)

        prompt = """
        これは統合報告書の最初の数ページの画像です。
        この中から会社名のみを抽出してください。

        - 「株式会社」「〇〇株式会社」形式が多い
        - 法人格を除いた会社名のみ出力
        - 補足や説明は禁止
        - 1行のみ
        - 判別できない場合は「取得失敗」
        """

        response = image_model.generate_content([prompt, *images])
        result = response.text.strip()
        return result if result else "取得失敗"

    except Exception as e:
        warnings.warn(f"Gemini画像処理失敗: {e}")
        return "取得失敗"


def update_組織名G(worksheet):
    logging.info("🏢 update_組織名G開始")

    df = get_as_dataframe(worksheet)
    df.fillna('', inplace=True)

    update_count = 0

    for idx, row in df.iterrows():
        url = row['URL']
        name_g = row.get('会社名G', '')
        page_count = row['ページ数']

        if not url or name_g:
            continue

        if isinstance(page_count, (int, float)) and page_count <= 15:
            df.at[idx, '会社名G'] = '対象外'
            update_count += 1
            logging.info(f"⏭️ 対象外: {url}")
            continue

        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            res = requests.get(url, headers=headers, timeout=15)

            if res.status_code == 200:
                extracted = extract_company_name_from_pdf_image(res.content)
                df.at[idx, '会社名G'] = extracted
                logging.info(f"🖼️ G抽出: {url} → {extracted}")
            else:
                df.at[idx, '会社名G'] = '取得失敗'
                logging.warning(f"⚠️ DL失敗 {url}")
        except Exception as e:
            df.at[idx, '会社名G'] = '取得失敗'
            logging.warning(f"❌ error: {e} {url}")

        update_count += 1

    # シートへ反映
    df.replace([np.nan, np.inf, -np.inf], '', inplace=True)
    col_index = df.columns.get_loc('会社名G')
    col_letter = chr(ord('A') + col_index)

    worksheet.update(
        f"{col_letter}2:{col_letter}{len(df)+1}",
        [[v] for v in df['会社名G'].tolist()]
    )

    logging.info(f"📄 {update_count} 件の会社名G更新")
    return f"{update_count} 件更新", 200


def update_組織名(worksheet):
    logging.info("🏢 update_組織名（T/G統合処理）開始")

    # Geminiモデルを初期化（共通関数）
    global text_model
    if text_model is None:
        text_model = init_gemini()

    df = get_as_dataframe(worksheet)
    df.fillna('', inplace=True)

    # 「会社名」列がなければ作成
    if '会社名' not in df.columns:
        df['会社名'] = ''

    update_count = 0

    # 無効値かどうかを判定する関数
    def is_invalid(name):
        return name in ['', '取得失敗', '対象外']

    for idx, row in df.iterrows():
        name_t = row.get('会社名T', '').strip()
        name_g = row.get('会社名G', '').strip()
        current = row.get('会社名', '').strip()

        # すでに決定済みはスキップ
        if current:
            continue

        # ------------------------
        # ① 両方無効 → 対象外
        # ------------------------
        if is_invalid(name_t) and is_invalid(name_g):
            df.at[idx, '会社名'] = '対象外'
            update_count += 1
            logging.info(f"⏭️ 対象外（両方無効）")
            continue

        # ------------------------
        # ② 片方のみ有効 → 採用
        # ------------------------
        if not is_invalid(name_t) and is_invalid(name_g):
            df.at[idx, '会社名'] = name_t
            update_count += 1
            logging.info(f"✅ 単独採用（T）: {name_t}")
            continue

        if not is_invalid(name_g) and is_invalid(name_t):
            df.at[idx, '会社名'] = name_g
            update_count += 1
            logging.info(f"✅ 単独採用（G）: {name_g}")
            continue

        # ------------------------
        # ③ 両方有効 → Gemini に判断させる
        # ------------------------
        try:
            prompt = f"""
            次の2つの会社名候補のうち、証券コードと紐付けしやすい
            「正式な会社名」として適切なものを1つ選んでください。

            - {name_t}
            - {name_g}

            条件:
            - より一般的・正式に見える名称を選ぶ
            - 選んだ名前のみ1行で返す（余計な説明禁止）
            """

            response = text_model.generate_content(prompt)
            best_name = response.text.strip()

            if best_name in [name_t, name_g]:
                df.at[idx, '会社名'] = best_name
                update_count += 1
                logging.info(f"🧠 Gemini判断: {best_name}")
            else:
                logging.warning(f"⚠️ Gemini判定不能: {best_name}")

        except Exception as e:
            logging.warning(f"Gemini判断失敗: {e}")

    # NaNを空文字へ
    df.replace([np.nan, np.inf, -np.inf], '', inplace=True)

    # シート更新（会社名列のみ）
    col_index = df.columns.get_loc('会社名')
    col_letter = chr(ord('A') + col_index)

    worksheet.update(
        f"{col_letter}2:{col_letter}{len(df)+1}",
        [[v] for v in df['会社名'].tolist()]
    )

    logging.info(f"📄 {update_count} 件の会社名を「会社名」列に更新")
    return f"{update_count} 件更新", 200
