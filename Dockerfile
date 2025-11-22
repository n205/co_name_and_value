FROM python:3.11-slim-bookworm

WORKDIR /app

# 必要な apt パッケージとフォントなど
RUN apt-get update && apt-get install -y \
    libssl-dev \
    libffi-dev \
    python3-dev \
    cargo \
    wget \
    unzip \
    curl \
    gnupg \
    ca-certificates \
    fonts-liberation \
    libappindicator3-1 \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libcups2 \
    libdbus-1-3 \
    libgdk-pixbuf2.0-0 \
    libnspr4 \
    libnss3 \
    libx11-xcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    xdg-utils \
    poppler-utils \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Chrome 137.0.7151.119 を直接インストール
RUN wget -q https://dl.google.com/linux/deb/pool/main/g/google-chrome-stable/google-chrome-stable_137.0.7151.119-1_amd64.deb \
 && apt-get update \
 && apt-get install -y ./google-chrome-stable_137.0.7151.119-1_amd64.deb \
 && rm google-chrome-stable_137.0.7151.119-1_amd64.deb

# 対応 ChromeDriver のインストール
RUN wget -q -O /tmp/chromedriver.zip \
    https://storage.googleapis.com/chrome-for-testing-public/137.0.7151.119/linux64/chromedriver-linux64.zip \
 && unzip /tmp/chromedriver.zip -d /tmp/ \
 && mv /tmp/chromedriver-linux64/chromedriver /usr/local/bin/chromedriver \
 && chmod +x /usr/local/bin/chromedriver \
 && rm -rf /tmp/chromedriver.zip /tmp/chromedriver-linux64

# Python ライブラリ関連
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . /app

CMD ["python", "main.py"]
