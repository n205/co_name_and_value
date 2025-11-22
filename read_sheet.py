import logging
import pandas as pd
import gspread
from gspread_dataframe import get_as_dataframe
from google.oauth2 import service_account
import time
import numpy as np

def read_sheet():
    SPREADSHEET_ID = '18Sb4CcAE5JPFeufHG97tLZz9Uj_TvSGklVQQhoFF28w'
    WORKSHEET_NAME = 'バリュー抽出'

    try:
        creds = service_account.Credentials.from_service_account_file(
            '/secrets/service-account-json',
            scopes=[
                'https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive'
            ]
        )
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(SPREADSHEET_ID)
        worksheet = sh.worksheet(WORKSHEET_NAME)

        existing_df = get_as_dataframe(worksheet).dropna(subset=['URL'])
        processed_urls = set(existing_df['URL'].tolist())

        logging.info(f'✅ 取得済URL数: {len(processed_urls)}')
        return worksheet, existing_df, processed_urls

    except Exception as e:
        import traceback
        logging.error('❌ エラー発生:\n' + traceback.format_exc())
        return f'エラー: {e}', 500
