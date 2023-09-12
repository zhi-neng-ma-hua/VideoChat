from sanic import Sanic
from sanic.response import text

from utils.logger_config import setup_logging, logger

# 初始化日志设置（Initialize logging settings）
setup_logging()


app = Sanic("video_chat_gpt")
# app.blueprint(api)

@app.get("/")
async def hello_world(request):
    logger("Received a request", "info")
    return text("Hello, world.")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)