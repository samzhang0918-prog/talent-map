FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 显式列出要拷贝的文件，而不是 COPY . .——仓库里没有需要排除的敏感文件，
# 但显式列出更清楚这个镜像实际跑的是哪些代码。
COPY app.py topic_graph.py ./
COPY lib ./lib

# Hugging Face Spaces 的 Docker SDK 默认期望容器监听 7860 端口；
# app.py 会读 PORT 环境变量，本地不设置时仍然默认 8000。
ENV PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
