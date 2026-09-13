from flask import Flask
import pandas as pd
import gspread
import time  # タイムアウト計測用に追加
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

# ★ Cloud Runの10分(600秒)制限に対し、余裕を持って8分30秒(510秒)を制限時間に設定
TIME_LIMIT = 510 

@app.route('/', methods=['GET', 'POST'])
def main():
    start_time = time.time()  # ★ 処理開始時刻を記録
    logging.info('📥 リクエスト受信')

    # スプレッドシート読込
    worksheet, existing_df, processed_urls = read_sheet()

    # タイムアウト対策対象外の処理（そのまま実行）
    update_組織名T(worksheet)
    update_組織名G(worksheet)
    update_組織名(worksheet)    
    update_証券番号(worksheet)

    # ★ タイムアウトをチェックするヘルパー関数
    def is_time_over():
        return (time.time() - start_time) > TIME_LIMIT

    # タイムアウト対策対象の処理（時間切れでなければ実行し、開始時間と制限時間を渡す）
    if not is_time_over():
        update_バリューT(worksheet, start_time, TIME_LIMIT)
    
    if not is_time_over():
        update_バリューG(worksheet, start_time, TIME_LIMIT)
    
    if not is_time_over():
        update_バリュー(worksheet, start_time, TIME_LIMIT)
    
    elapsed_time = int(time.time() - start_time)
    logging.info(f'✅ 処理完了 (所要時間: {elapsed_time}秒)')

    return 'Cloud Run Function executed.', 200


if __name__ == '__main__':
    logging.info('🚀 アプリ起動')
    app.run(host='0.0.0.0', port=8080)
