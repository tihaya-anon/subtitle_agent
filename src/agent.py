# 文件: src/agent.py

import os
import sys
import glob
import subprocess
import srt
import json
import time
from pathlib import Path

# --- 导入新模块 ---
from src.logger import logger
from config import settings
from src.core.llm_processor import process_with_llm
# --------------------

# --- [函数已修改] ---
def run_asr_and_diarization(input_wav: Path, output_srt: Path) -> bool:
    """
    步骤 1：调用 src/core/diarization.py (基于 wespeaker 的版本)。
    (已更新以捕获 stderr)
    """
    logger.info(f"1. 开始 ASR 和 Diarization: {input_wav.name}...")
    
    if not settings.DIARIZATION_SCRIPT_PATH.exists():
        logger.error(f"未找到 diarization 脚本: {settings.DIARIZATION_SCRIPT_PATH}")
        return False
        
    command = [
        sys.executable,  # 使用当前的 python 解释器
        str(settings.DIARIZATION_SCRIPT_PATH),
        '--input_file', str(input_wav),
        '--output_file', str(output_srt),
        '--whisper_model_type', settings.WHISPER_MODEL_TYPE,
        '--embedding_model', settings.EMBEDDING_MODEL,
        '--language', settings.TARGET_LANGUAGE
    ]
    
    try:
        # [修改] 添加 capture_output=True 来捕获 stdout 和 stderr
        process = subprocess.run(
            command, 
            text=True, 
            encoding='utf-8', 
            check=True, 
            capture_output=True # <--- 关键修改
        )
        
        # [新] 即使成功，也打印 stdout (whisper 的进度)
        logger.info(f"[diarization.py STDOUT]:\n{process.stdout}")
            
        if not output_srt.exists():
            logger.error(f"'diarization.py' 成功运行，但未创建SRT文件: {output_srt}")
            return False
            
        logger.info(f"ASR 成功。SRT 已保存到: {output_srt}")
        return True
        
    except subprocess.CalledProcessError as e:
         # [修改] 捕获错误时，打印详细的 stderr
         logger.error(f"'diarization.py' 执行失败，返回码: {e.returncode}")
         logger.error(f"--- [diarization.py 错误日志 STDERR] ---")
         logger.error(f"\n{e.stderr}") # <--- 关键修改
         logger.error(f"--- [diarization.py 错误日志 END] ---")
         return False
    except Exception as e:
        logger.error(f"'diarization.py' 启动时发生意外错误: {e}", exc_info=True)
        return False
# --- [函数修改结束] ---


def parse_srt(srt_file: Path) -> list:
    """
    步骤 2：解析 .srt 文件，提取任务队列。
    """
    logger.info(f"2. 解析 SRT 文件: {srt_file.name}...")
    try:
        with open(srt_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        subs = list(srt.parse(content))
        if not subs:
            logger.warning(f"SRT 文件为空: {srt_file.name}")
            return []
        subtitle_queue = []
        for sub in subs:
            subtitle_queue.append({
                "start_time": srt.timedelta_to_srt_timestamp(sub.start),
                "end_time": srt.timedelta_to_srt_timestamp(sub.end),
                "original_text_with_speaker": sub.content.replace('\n', ' ')
            })
        logger.info(f"解析成功，共 {len(subtitle_queue)} 条台词待处理。")
        return subtitle_queue
    except Exception as e:
        logger.error(f"解析 SRT 文件失败: {e}", exc_info=True)
        return []

def run_single_file_process(input_wav_path: Path, anime_name: str) -> (Path | None):
    """
    自动化 Agent 单文件处理主流程。
    如果成功，返回最终 JSON 文件的路径。
    """
    logger.info(f"\n=======================================================")
    logger.info(f"=== 开始处理文件: {input_wav_path.name} ===")
    logger.info(f"=== 使用指定动漫上下文: {anime_name if anime_name else '自动检测'}") 
    
    if not settings.DEEPSEEK_API_KEY:
        logger.critical("致命错误: 环境变量 'DEEPSEEK_API_KEY' 未设置。请检查 .env 文件。")
        return None

    # 确保文件夹存在
    settings.INTERMEDIATE_SRT_FOLDER.mkdir(parents=True, exist_ok=True)
    settings.FINAL_OUTPUT_JSON_FOLDER.mkdir(parents=True, exist_ok=True)
    
    file_basename = input_wav_path.stem
    intermediate_srt_path = settings.INTERMEDIATE_SRT_FOLDER / f"{file_basename}_raw.srt"
    final_output_json_path = settings.FINAL_OUTPUT_JSON_FOLDER / f"{file_basename}_structural.json"

    # --- 步骤 1: ASR + Diarization ---
    if not run_asr_and_diarization(input_wav_path, intermediate_srt_path):
        logger.error(f"ASR 失败: {input_wav_path.name}")
        return None

    # --- 步骤 2: 解析 SRT ---
    subtitle_queue = parse_srt(intermediate_srt_path)
    if not subtitle_queue:
        logger.error(f"SRT 解析失败或文件为空: {input_wav_path.name}")
        return None
        
    # --- 步骤 3 & 4: LLM 处理与结果汇总 ---
    logger.info(f"3. 开始使用 LLM 处理 {len(subtitle_queue)} 条台词...")
    final_data = []
    
    for i, item in enumerate(subtitle_queue, 1):
        logger.info(f"--- 正在处理 {file_basename} 的第 {i}/{len(subtitle_queue)} 条 ---")
        
        llm_result = process_with_llm(item['original_text_with_speaker'], anime_name)
        
        if llm_result:
            final_entry = {
                "start_time": item['start_time'],
                "end_time": item['end_time'],
                "speaker_id": llm_result.get('speaker_id', 'unknown'),
                "dialogue_content": llm_result.get('translated_text', '翻译失败'),
                "emotion_label": llm_result.get('emotion_type', '未知情感'),
                "emotion_intensity": llm_result.get('emotion_intensity', 0)
            }
            final_data.append(final_entry)
        else:
            logger.warning(f"第 {i} 条台词处理失败，跳过。")
        time.sleep(1) # API 限速

    # --- 步骤 5: 保存最终文件 ---
    logger.info(f"4. 文件 {file_basename} 处理完毕。正在保存到 {final_output_json_path}...")
    try:
        with open(final_output_json_path, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)
        logger.info(f"5. 成功！文件 {final_output_json_path} 已保存。")
        return final_output_json_path # <-- 成功后返回路径
    except IOError as e:
        logger.error(f"写入 JSON 文件失败: {e}", exc_info=True)
        return None

def run_batch_process():
    """
    自动化 Agent 批量处理主流程。
    """
    logger.info("--- 字幕预处理 Agent (批量模式) 已启动 ---")
    
    settings.INPUT_WAV_FOLDER.mkdir(parents=True, exist_ok=True)
    wav_files_to_process = list(settings.INPUT_WAV_FOLDER.glob('*.wav'))
    
    if not wav_files_to_process:
        logger.warning(f"在 '{settings.INPUT_WAV_FOLDER}' 中未找到任何 .wav 文件。")
        return
        
    logger.info(f"发现 {len(wav_files_to_process)} 个 .wav 文件。开始批量处理...")
    
    auto_detect_anime = ""
    
    for input_wav_path in wav_files_to_process:
        run_single_file_process(input_wav_path, auto_detect_anime)
    
    logger.info("\n=======================================================")
    logger.info("--- 字幕预处理 Agent (批量模式) 任务完成 ---")