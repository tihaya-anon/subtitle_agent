import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件 (会查找同级或父级的 .env)
load_dotenv()

# --- 1. 根路径 ---
# (Path(__file__) -> settings.py, .parent -> config/, .parent -> SubtitleAgent/)
BASE_DIR = Path(__file__).resolve().parent.parent

# --- 2. API 配置 ---
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

# --- 3. 数据文件夹路径 ---
DATA_DIR = BASE_DIR / "data"
INPUT_WAV_FOLDER = DATA_DIR / "input/wav"
INTERMEDIATE_SRT_FOLDER = DATA_DIR / "intermediate/srt"
FINAL_OUTPUT_JSON_FOLDER = DATA_DIR / "final/json"

# --- 4. 脚本和模型配置 ---
# 注意：现在 diarization.py 在 src/core/ 目录下
DIARIZATION_SCRIPT_PATH = BASE_DIR / "src/core/diarization.py"
WHISPER_MODEL_TYPE = "medium"
EMBEDDING_MODEL = "wespeaker"
TARGET_LANGUAGE = "ja"

# --- 5. 日志配置 ---
LOG_FILE_PATH = BASE_DIR / "app.log"