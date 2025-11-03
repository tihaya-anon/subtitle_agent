# 文件: src/core/diarization.py

# 确保在导入 src 模块之前，src 的父目录 (项目根目录) 在 Python 路径上
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import List, Optional, Tuple, Dict, NamedTuple
import argparse
import os
import tempfile # <--- [V5 修复] 导入 tempfile

import srt
import numpy as np
import tqdm
import whisper  
import torchaudio
import torch # <--- [V5 修复] 移除了不再需要的 T
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.diarization_utils import get_wespeaker_model, write_subtitle  


device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Segment(NamedTuple):
    start: float
    end: float
    text: str
    speaker: str = "unknown"


def parse_exemplar_dir(exemplar_dir: Optional[str]) -> Optional[Dict[str, List[str]]]:
    """Parse exemplar directory into a mapping of speaker names to their audio files."""
    if not exemplar_dir:
        return None
    return {
        spk_dir: [os.path.join(exemplar_dir, spk_dir, f) for f in os.listdir(os.path.join(exemplar_dir, spk_dir))]
        for spk_dir in os.listdir(exemplar_dir)
    }


# [!!] 关键修改: 这是被重写的 transcribe 函数 (使用 openai-whisper)
def transcribe(
    input_file: str,
    model_type: str = "medium",
    language: Optional[str] = None,
) -> List[Segment]:
    """
    Transcribe audio and return segments with timestamps and text.
    (使用官方 openai-whisper 库, 绕过 whisperx 和 pyannote.audio 的依赖冲突)
    """
    print(f"[transcribe] 正在加载 Whisper 模型: {model_type}...")
    
    # 1. 加载模型
    model = whisper.load_model(model_type, device=device)
    
    print(f"[transcribe] 正在转录音频: {input_file}...")
    
    # 2. 转录音频
    fp16_option = device == "cuda"
    result = model.transcribe(input_file, language=language, fp16=fp16_option)
    
    print("[transcribe] 转录完成。正在格式化分段...")

    # 4. 格式化输出
    segments_list = [
        Segment(s["start"], s["end"], s["text"].strip())
        for s in result["segments"]
        if s["end"] - s["start"] > 0
    ]
    
    print(f"[transcribe] 共找到 {len(segments_list)} 个分段。")
    return segments_list


def extract_embeddings(
    segments: List[Segment],
    input_file: str,
    exemplars: Optional[Dict[str, List[str]]] = None,
    model_type: str = "wespeaker",
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[List[str]]]:
    """Extract speaker embeddings using either WeSpeaker or ECAPA-TDNN model."""
    signal, fs = torchaudio.load(input_file)
    signal = signal.to(device) # <--- 音频数据在这里被移到 'mps'
    
    # <--- [V5 修复] 移除重采样逻辑 (torchaudio.save 会处理)
    
    # Initialize model
    if model_type == "ecapatdnn":
        from speechbrain.inference.speaker import EncoderClassifier
        model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device}
        )
        def get_embedding(audio):
            # ECAPA-TDNN 逻辑 (保持 V3 修复)
            audio_batch = audio.unsqueeze(0)
            embedding_batch = model.encode_batch(audio_batch)
            return np.squeeze(embedding_batch.detach().cpu().numpy())
    else:  # wespeaker
            # <--- [V5 修复] 还原到你最初的、使用 tempfile 的正确逻辑
            import wespeaker
            
            # 1. 加载高级 API 模型
            speaker_model = wespeaker.load_model(get_wespeaker_model())
            
            def get_embedding(audio):
                # audio 是一个 [C, T] 的 Tensor (在 'mps' 上)
                
                # 2. 将音频移到 CPU 以便保存
                audio_cpu = audio.cpu()

                # 确保是单声道 (wespeaker 可能需要)
                if audio_cpu.dim() > 1 and audio_cpu.shape[0] > 1:
                    audio_cpu = audio_cpu[0:1, :]
                
                tmp_path = "" # 初始化
                try:
                    # 3. 创建一个临时 .wav 文件
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                        tmp_path = tmp_file.name
                    
                    # 4. 将 Tensor 切片保存到临时文件中 (使用原始采样率 fs)
                    torchaudio.save(tmp_path, audio_cpu, fs)
                    
                    # 5. [关键] 调用高级 API，传入文件路径
                    # wespeaker 内部会处理加载、重采样、和设备(cpu/gpu)
                    embedding = speaker_model.extract_embedding(tmp_path)
                    
                    return embedding.detach().cpu().numpy()
                
                finally:
                    # 6. 清理临时文件
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
            # <--- [V5 修复结束] ---

    
    # Extract embeddings for segments
    embeddings = []
    for seg in tqdm.tqdm(segments, desc="Extracting embeddings"):
        # audio_seg 也是 'mps' 上的
        audio_seg = signal[:, int(seg.start * fs):int(seg.end * fs)]
        if audio_seg.shape[1] < fs * 0.5:  # Pad short segments
            audio_seg = torch.nn.functional.pad(audio_seg, (0, int(fs * 0.5) - audio_seg.shape[1]))
        
        # 将 'mps' 上的 Tensor 传递给 get_embedding
        embeddings.append(get_embedding(audio_seg))
    
    if not embeddings:
        print("[extract_embeddings] 警告: 没有从分段中提取到任何 embedding。")
        empty_np = np.empty((0, 0)) 
        return empty_np, None, None

    embeddings = np.stack(embeddings, axis=0)
    
    # Process exemplars if provided
    if not exemplars:
        return embeddings, None, None
        
    embeddings_exemplar = []
    spk_list = []
    for spk_name, spk_files in exemplars.items():
        spk_embeddings = []
        for spk_file in spk_files:
            # [V5 修复] 此处 exemplar 已经是文件路径，
            # 我们不需要加载它，直接传递给 get_embedding_from_path
            # (为了保持一致，我们重用 get_embedding，它会加载并保存)
            signal_ex, fs_ex = torchaudio.load(spk_file)
            signal_ex = signal_ex.to(device) # <--- exemplar 音频移到 mps
            
            # 如果采样率不一致
            if fs_ex != fs:
                print(f"警告: Exemplar 采样率 {fs_ex} 与主音频 {fs} 不匹配，正在重采样...")
                resampler = torchaudio.transforms.Resample(fs_ex, fs).to(device)
                signal_ex = resampler(signal_ex)
                    
            spk_embeddings.append(get_embedding(signal_ex)) # <--- 重用 get_embedding
        embeddings_exemplar.append(np.mean(spk_embeddings, axis=0))
        spk_list.append(spk_name)
    
    return embeddings, np.stack(embeddings_exemplar, axis=0), spk_list


def cluster_segments(
    segments: List[Segment],
    embeddings: np.ndarray,
    n_cluster: Optional[int] = None,
    distance_threshold: float = 0.8,
) -> List[Segment]:
    """Cluster segments by speaker embeddings."""
    if len(embeddings) < 2:
        if len(embeddings) == 0:
            print("[cluster_segments] 警告: 传入了 0 个 embeddings, 无法聚类。")
            return segments # 返回原始分段
        return [Segment(seg.start, seg.end, seg.text, "spk_0") for seg in segments]
        
    clustering = AgglomerativeClustering(
        n_clusters=n_cluster,
        metric="cosine",
        linkage="average",
        distance_threshold=None if n_cluster else distance_threshold
    ).fit_predict(embeddings)
    
    return [
        Segment(seg.start, seg.end, seg.text, f"spk_{clustering[i]}")
        for i, seg in enumerate(segments)
    ]


def assign_speakers(
    segments: List[Segment],
    embeddings: np.ndarray,
    embeddings_exemplar: np.ndarray,
    spk_list: List[str],
    threshold: float,
) -> List[Segment]:
    """Assign speakers to segments based on exemplar embeddings."""
    if len(embeddings) == 0:
        print("[assign_speakers] 警告: 传入了 0 个 embeddings, 无法分配说话人。")
        return segments
        
    similarity = cosine_similarity(embeddings, embeddings_exemplar)
    labels = np.argmax(similarity, axis=1)
    max_sims = np.max(similarity, axis=1)
    print(labels, max_sims)
    return [
        Segment(seg.start, seg.end, seg.text, spk_list[labels[i]] if max_sims[i] > threshold else "unknown")
        for i, seg in enumerate(segments)
    ]


def main(args: argparse.Namespace) -> None:
    """Main entry point for diarization and subtitle generation."""
    # Transcribe audio
    segments = transcribe(args.input_file, args.whisper_model_type, args.language)
    
    if not segments:
        print(f"转录失败或音频为空，未在 '{args.input_file}' 中找到任何台词。")
        # 确保目录存在
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        write_subtitle([], out_file=args.output_file)
        print(f"已创建空的 srt 文件: {args.output_file}")
        return # 提前退出

    # Get exemplars if provided
    exemplars = parse_exemplar_dir(args.exemplar_dir)
    
    # Extract embeddings
    embeddings, embeddings_exemplar, spk_list = extract_embeddings(
        segments,
        args.input_file,
        exemplars,
        args.embedding_model
    )
    
    if embeddings.size == 0:
        print("提取 Embeddings 失败。将只输出无说话人ID的字幕。")
        segments_with_default_spk = [
             Segment(seg.start, seg.end, seg.text, "spk_0")
             for seg in segments
        ]
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        write_subtitle(segments_with_default_spk, out_file=args.output_file)
        print(f"已保存（无说话人分离）的 srt 文件: {args.output_file}")
        return # 提前退出

    # Process segments
    if exemplars:
        segments = assign_speakers(
            segments,
            embeddings,
            embeddings_exemplar,
            spk_list,
            args.exemplar_threshold
        )
    else:
        segments = cluster_segments(
            segments,
            embeddings,
            args.n_cluster,
            args.distance_threshold
        )
    
    # Write subtitle
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    write_subtitle(segments, out_file=args.output_file)
    print(f"成功！带说话人ID的 srt 文件已保存: {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speaker diarization using Whisper and speaker embeddings")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input audio file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output srt file")
    parser.add_argument("--whisper_model_type", type=str, default="medium", help="Type of whisper model")
    parser.add_argument("--language", type=str, default=None, help="Language code for transcription")
    parser.add_argument("--embedding_model", type=str, default="wespeaker", choices=["ecapatdnn", "wespeaker"],
                       help="Type of speaker embedding model")
    parser.add_argument("--n_cluster", type=int, default=None, help="Number of speakers")
    parser.add_argument("--distance_threshold", type=float, default=0.8,
                       help="Distance threshold for clustering when n_cluster is None")
    parser.add_argument("--exemplar_dir", type=str, default=None, help="Path to audio exemplars")
    parser.add_argument("--exemplar_threshold", type=float, default=0.2,
                       help="Threshold for assigning unknown speakers (cosine similarity)")
    
    main(parser.parse_args())