FROM python:3.11-slim

# Python 对非 tty 的 stdout 是整块缓冲的，容器里不加这个的话 print() 打的诊断
# 日志可能要等缓冲区攒满才会出现在平台的日志面板里，公开部署排障时非常致命。
ENV PYTHONUNBUFFERED=1

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
