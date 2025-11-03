# Subtitle Agent (字幕处理工具)

`SubtitleAgent` 是一个桌面应用程序，旨在自动化从音频文件中提取、翻译和结构化字幕的完整流程。

它集成了语音识别 (ASR)、说话人日志 (Diarization) 和大型语言模型 (LLM) 翻译，通过一个简洁的图形用户界面 (GUI) 提供高质量、上下文感知的字幕翻译。

## 核心功能

  * **ASR 与说话人日志**: 使用 `openai-whisper` 进行高精度语音转录，并结合 `wespeaker` 识别不同说话人 (spk\_0, spk\_1...)。
  * **上下文感知翻译**: 利用 `DeepSeek` API，不仅翻译台词，还能进行情感分析。
  * **指定动漫上下文**: 用户可以在 GUI 中输入动漫的中文名称（例如："BanG Dream\! It's MyGO\!\!\!\!\!"）。AI 将调用其知识库，强制使用该动漫的官方或通用译名（例如：`クライシック` -\> `Crychic`），极大提高了专有名词的准确性。
  * **图形用户界面**: 使用 `PySide6` 构建，提供文件浏览、实时日志显示和最终结果表格等功能。
  * **结构化输出**: 最终结果保存为 `.json` 文件，包含时间戳、说话人ID、翻译内容、情感标签和情感强度。

## 快速上手

### 1\. 环境配置

本项目依赖 Python 3.10+ 和 `ffmpeg`。

**安装 ffmpeg:**

  * **macOS (使用 Homebrew):**
    ```bash
    brew install ffmpeg
    ```
  * **Windows (使用 winget):**
    ```bash
    winget install -e --id Gyan.FFmpeg
    ```

### 2\. 项目设置

1.  **克隆或下载项目:**
    (假设你已将项目文件放在 `SubtitleAgent` 目录中)

2.  **进入项目目录:**

    ```bash
    cd path/to/SubtitleAgent
    ```

3.  **安装依赖:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **配置 API 密钥 (关键):**

      * 在项目根目录创建一个名为 `.env` 的文件。
      * 打开该文件，填入你的 DeepSeek API 密钥：

    <!-- end list -->

    ```
    DEEPSEEK_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxx"
    ```

### 3\. 运行程序

在项目根目录 (`SubtitleAgent/`) 下运行：

```bash
python main.py
```

### 4\. 如何使用

1.  程序启动后，会显示 GUI 界面。
2.  点击 **"浏览..."** 按钮，选择一个 `.wav` 音频文件。
3.  **(推荐)** 在 **"指定动漫"** 输入框中，填入该音频所属动漫的中文全名。
4.  点击 **"开始处理"** 按钮。
5.  在 **"运行日志"** 窗口中观察实时处理进度。
6.  处理完成后，结果将自动填充到 **"处理结果"** 表格中。
7.  点击 **"打开输出目录"** 可查看最终生成的 `.json` 文件 (位于 `data/final/json/`)。