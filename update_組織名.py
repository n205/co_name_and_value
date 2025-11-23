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
import gc


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

    reader = None
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        all_text = ""

        for i in range(min(3, len(reader.pages))):
            text = reader.pages[i].extract_text()
            if text:
                all_text += text + "\n"

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

    finally:
        # ---- メモリ解放 ----
        del reader
        del pdf_bytes
        gc.collect()


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

    images = None
    try:
        images = convert_from_bytes(pdf_bytes, dpi=150, first_page=1, last_page=3)

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

    finally:
        # ---- 画像メモリ解放 ----
        if images:
            for img in images:
                try:
                    img.close()
                except:
                    pass
            del images

        del pdf_bytes
        gc.collect()


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

    # シート更新
    df.replace([np.nan, np.inf, -np.inf], '', inplace=True)
    col_index = df.columns.get_loc('会社名G')
    col_letter = chr(ord('A') + col_index)

    worksheet.update(
        f"{col_letter}2:{col_letter}{len(df)+1}",
        [[v] for v in df['会社名G'].tolist()]
    )

    logging.info(f"📄 {update_count} 件の会社名G更新")
    return f"{update_count} 件更新", 200


# ============================================================
#  3) T/G統合 → 会社名
# ============================================================
def update_組織名(worksheet):
    logging.info("🏢 update_組織名（T/G統合処理）開始")

    global text_model
    if text_model is None:
        text_model = init_gemini()

    df = get_as_dataframe(worksheet)
    df.fillna('', inplace=True)

    # 「会社名」列がなければ作成
    if '会社名' not in df.columns:
        df['会社名'] = ''

    update_count = 0

    def is_invalid(name):
        return name in ['', '取得失敗', '対象外']

    for idx, row in df.iterrows():
        name_t = row.get('会社名T', '').strip()
        name_g = row.get('会社名G', '').strip()
        current = row.get('会社名', '').strip()

        if current:
            continue

        if is_invalid(name_t) and is_invalid(name_g):
            df.at[idx, '会社名'] = '対象外'
            update_count += 1
            logging.info("⏭️ 対象外（両方無効）")
            continue

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

        # 両方有効 → Gemini 判定
        try:
            prompt = f"""
            次の2つの会社名候補のうち、
            より正式な会社名として適切なものを選んでください。

            - {name_t}
            - {name_g}

            条件:
            - 選んだ名前のみ1行で返す
            """

            response = text_model.generate_content(prompt)
            best_name = response.text.strip()

            if best_name in [name_t, name_g]:
                df.at[idx, '会社名'] = best_name
                update_count += 1
                logging.info(f"🧠 Gemini判断: {best_name}")
            else:
                logging.warning(f"⚠️ 判定不能: {best_name}")

        except Exception as e:
            logging.warning(f"Gemini判断失敗: {e}")

    # シート更新
    df.replace([np.nan, np.inf, -np.inf], '', inplace=True)
    col_index = df.columns.get_loc('会社名')
    col_letter = chr(ord('A') + col_index)

    worksheet.update(
        f"{col_letter}2:{col_letter}{len(df)+1}",
        [[v] for v in df['会社名'].tolist()]
    )

    logging.info(f"📄 {update_count} 件の会社名を更新")
    return f"{update_count} 件更新", 200


# ============================================================
#  4) 証券番号推定
# ============================================================
def update_証券番号(worksheet):
    logging.info("💹 update_証券番号開始")

    global text_model
    if text_model is None:
        text_model = init_gemini()

    df = get_as_dataframe(worksheet)
    df.fillna('', inplace=True)

    if '証券番号' not in df.columns:
        df['証券番号'] = ''

    update_count = 0

    for idx, row in df.iterrows():
        company = row.get("会社名", "").strip()
        current_code = row.get("証券番号", "").strip()

        if current_code:
            continue

        if company in ["対象外", "取得失敗", ""]:
            df.at[idx, "証券番号"] = "対象外"
            update_count += 1
            logging.info(f"⏭️ 対象外扱い: {company}")
            continue

        try:
            prompt = f"""
            以下の会社名から日本の証券コード（4桁）を推定してください。

            条件:
            - 出力は4桁のみ
            - 存在しない場合は「対象外」
            - 補足説明禁止

            会社名: {company}
            """

            response = text_model.generate_content(prompt)
            code = response.text.strip()

            if code.isdigit() and len(code) == 4:
                df.at[idx, "証券番号"] = code
                update_count += 1
                logging.info(f"✅ {company} → {code}")
            else:
                df.at[idx, "証券番号"] = "対象外"
                update_count += 1
                logging.info(f"⚠️ 不明 → 対象外: {company} → {code}")

        except Exception as e:
            df.at[idx, "証券番号"] = "対象外"
            update_count += 1
            logging.warning(f"❌ エラー → 対象外扱い: {e}")

    # シート更新
    df.replace([np.nan, np.inf, -np.inf], '', inplace=True)

    def column_index_to_letter(index):
        letters = ""
        while index >= 0:
            index, remainder = divmod(index, 26)
            letters = chr(65 + remainder) + letters
            index -= 1
        return letters

    col_index = df.columns.get_loc("証券番号")
    col_letter = column_index_to_letter(col_index)

    worksheet.update(
        f"{col_letter}2:{col_letter}{len(df)+1}",
        [[v] for v in df["証券番号"].tolist()]
    )

    logging.info(f"📄 {update_count} 件の証券番号を更新")
    return f"{update_count} 件更新", 200
