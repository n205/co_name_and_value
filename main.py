from flask import Flask
import pandas as pd
import gspread
from gspread_dataframe import get_as_dataframe
from google.oauth2 import service_account
import logging

from read_sheet import read_sheet
from update_組織名 import update_組織名T
from update_組織名 import update_組織名G
from update_組織名 import update_組織名
from update_組織名 import update_証券番号
from update_価値ある活動 import update_バリューT
from update_価値ある活動 import update_バリューG
from update_価値ある活動 import update_バリュー


# Cloud Logging に出力するよう設定
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main():
    logging.info('📥 リクエスト受信')

    # スプレッドシート読込
    worksheet, existing_df, processed_urls = read_sheet()

    update_組織名T(worksheet)
    update_組織名G(worksheet)
    update_組織名(worksheet)    
    update_証券番号(worksheet)
    update_バリューT(worksheet)
    update_バリューG(worksheet)
    update_バリュー(worksheet)
    
    return 'Cloud Run Function executed.', 200


if __name__ == '__main__':
    logging.info('🚀 アプリ起動')
    app.run(host='0.0.0.0', port=8080)
