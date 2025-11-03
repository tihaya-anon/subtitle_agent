# 文件路径: src/core/llm_processor.py

import json
import requests
from src.logger import logger  # <-- 使用新的 logger
from config import settings     # <-- 从配置中导入 API 信息

def process_with_llm(text_with_speaker: str, anime_name: str):
    """
    步骤 3：调用 DeepSeek API 进行翻译、润色和情感标注。
    (V5 - 接收并强制使用用户指定的动漫上下文)
    """
    logger.info(f"正在调用 LLM (V5 Prompt) 处理: '{text_with_speaker[:40]}...'")
    
    if not settings.DEEPSEEK_API_KEY:
        logger.error("DEEPSEEK_API_KEY 未在 config/settings.py 或 .env 中设置。")
        return None

    headers = {
        'Authorization': f'Bearer {settings.DEEPSEEK_API_KEY}',
        'Content-Type': 'application/json'
    }

    # --- [新功能] 根据 anime_name 动态设置 Prompt ---
    if anime_name:
        # 如果用户指定了动漫
        context_instruction = f"你必须严格以【{anime_name}】这部作品的官方中文翻译和通用约定为准则进行翻译。"
        system_prompt = f"你是一个专业的动漫翻译师，当前的任务是翻译【{anime_name}】的台词。请严格按照JSON格式返回。"
    else:
        # 如果用户未指定，让 AI 自动检测
        context_instruction = "你必须自动检测台词来自哪部动漫，并使用该作品的官方中文翻译和通用约定为准则。"
        system_prompt = "你是一个精通ACG文化、拥有海量动漫知识的专业翻译。请严格按照JSON格式返回。"
    # --- 结束 ---


    # --- Prompt 修改开始 (V5) ---
    prompt_content = f"""
    你是一个精通ACG文化、拥有海量动漫知识的专业翻译和情感分析师。请严格按照JSON格式返回你的分析结果。

    请分析以下日文台词（台词内容已包含说话人ID，格式为 "spk_X : ..."）：
    "{text_with_speaker}"
    
    请严格执行以下任务：
    
    **第一步 (上下文准则)：**
    {context_instruction}
    
    **第二步 (执行翻译与分析)：**
    1.  **提取说话人ID** (例如 "spk_0") 。
    2.  **翻译台词内容 (关键要求)**：
        * 你必须调用你的动漫知识库，严格按照【第一步】的上下文准则，来处理所有【专有名词】、【人名】、【地名】、【团队名】、【技能名】和【特殊术语】等等。
        * **绝不能**对这些专有名词进行字面上的“直译”。
    
    **【翻译范例 - 你必须学习这个模式】：**
    * **例 1 (乐队名):** 日文 `クライシック` (Kuraishikku)，如果上下文是《BanG Dream!》，必须翻译为乐队名 "Crychic"，**绝不能**翻译为 "经典"。
    * **例 2 (人名):** 日文 `エレン` (Eren)，如果上下文是《进击的巨人》，必须翻译为 "艾伦"，**绝不能**翻译为 "埃伦" 或 "艾连"。
    * **例 3 (技能名):** 日文 `螺旋丸` (Rasengan)，如果上下文是《火影忍者》，必须翻译为 "螺旋丸"，**绝不能**翻译为 "旋转球" 或 "螺旋球"。
    
    3.  **润色翻译结果**：在严格遵守专有名词规则的基础上，将整句台词润色为口语化的简体中文。
    4.  **标注情感类型** (从 [\"平静\", \"开心\", \"悲伤\", \"愤怒\", \"惊讶\", \"恐惧\", \"厌恶\"] 中选择)。
    5.  **标注情感强度** (1-5，1为最弱，5为最强)。
    
    **第三步 (格式化输出)：**
    请严格按照此JSON格式返回，不要包含任何额外的解释：
    {{
      "speaker_id": "（提取的说话人ID）",
      "translated_text": "（你基于【指定动漫上下文】和作品知识，翻译并润色的结果）",
      "emotion_type": "（你选择的情感）",
      "emotion_intensity": （一个1到5的数字）
    }}
    """
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_content}
        ],
        "response_format": {"type": "json_object"}
    }
    
    try:
        response = requests.post(settings.DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        llm_data = response.json()
        content_string = llm_data['choices'][0]['message']['content']
        result_data = json.loads(content_string)
        return result_data
    except Exception as e:
        logger.error(f"LLM API 请求失败或解析 JSON 失败: {e}", exc_info=True)
    
    return None