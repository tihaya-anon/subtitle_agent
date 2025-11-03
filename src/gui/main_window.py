# 文件: src/gui/main_window.py

import sys
import os
import json
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QApplication, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QTextEdit, QGroupBox, QFormLayout, QLineEdit,
    QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QLabel, QSplitter
)
from PySide6.QtCore import QThread, Signal, QObject, QUrl, Qt
from PySide6.QtGui import QDesktopServices
import logging

# 导入我们的核心 Agent 逻辑和 logger
from src.agent import run_single_file_process
from src.logger import logger
from config import settings

# --- [修改] 后台线程 ---
class AgentThread(QThread):
    """
    修改后的线程，接受文件路径和动漫名称
    """
    finished_with_result = Signal(str)

    def __init__(self, input_path, anime_name): # <--- [修改] 接收 anime_name
        super().__init__()
        self.input_path = Path(input_path)
        self.anime_name = anime_name               # <--- [修改] 存储 anime_name

    def run(self):
        logger.info("Agent 线程已启动...")
        try:
            # [修改] 调用单文件处理器，传入动漫名称
            output_json_path = run_single_file_process(self.input_path, self.anime_name)
            
            if output_json_path:
                logger.info(f"Agent 线程处理完毕。结果: {output_json_path}")
                self.finished_with_result.emit(str(output_json_path))
            else:
                logger.error("Agent 线程处理失败，未返回结果路径。")
                self.finished_with_result.emit(None) # 发出 None 表示失败

        except Exception as e:
            logger.error(f"Agent 线程发生错误: {e}", exc_info=True)
            self.finished_with_result.emit(None) # 发出 None 表示失败

# --- 日志 Handler (保持不变) ---
class QtLogHandler(logging.Handler, QObject):
    new_log_message = Signal(str)
    
    def __init__(self):
        logging.Handler.__init__(self)
        QObject.__init__(self)

    def emit(self, record):
        message = self.format(record)
        self.new_log_message.emit(message.strip())


# --- [重构] 主窗口 ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Subtitle Agent v1.2 (指定动漫上下文)")
        self.setGeometry(100, 100, 900, 700) # 调小窗口

        self.input_file_path = None
        self.agent_thread = None

        # --- 1. 创建主布局 (垂直) ---
        main_layout = QVBoxLayout()
        
        # --- 2. 顶部区域 (文件选择 + 按钮) ---
        top_layout = QHBoxLayout()
        
        # 2a. 文件选择组
        file_group = QGroupBox("文件与上下文") # <--- [修改] 标题
        form_layout = QFormLayout()
        
        # 音频文件
        self.le_audio_file = QLineEdit()
        self.le_audio_file.setReadOnly(True)
        btn_browse_audio = QPushButton("浏览...")
        btn_browse_audio.clicked.connect(self.browse_audio_file)
        
        audio_file_layout = QHBoxLayout()
        audio_file_layout.addWidget(self.le_audio_file)
        audio_file_layout.addWidget(btn_browse_audio)
        
        form_layout.addRow("音频文件 (.wav):", audio_file_layout)
        
        # --- [新功能] 指定动漫名称 ---
        self.le_anime_name = QLineEdit()
        self.le_anime_name.setPlaceholderText("（可选）输入中文动漫名，如：BanG Dream! It's MyGO!!!!!")
        form_layout.addRow("指定动漫:", self.le_anime_name)
        # --- 结束 ---
        
        file_group.setLayout(form_layout)
        
        # 2b. 控制按钮组
        control_group = QGroupBox("控制")
        control_layout = QVBoxLayout()
        self.btn_start = QPushButton("开始处理")
        self.btn_stop = QPushButton("停止处理 (未实现)")
        self.btn_open_output = QPushButton("打开输出目录")
        self.btn_clear_log = QPushButton("清空日志")
        
        self.btn_start.clicked.connect(self.run_process)
        self.btn_stop.setEnabled(False) 
        self.btn_open_output.clicked.connect(self.open_output_dir)
        self.btn_clear_log.clicked.connect(self.clear_log)
        
        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(self.btn_stop)
        control_layout.addWidget(self.btn_open_output)
        control_layout.addWidget(self.btn_clear_log)
        control_group.setLayout(control_layout)

        top_layout.addWidget(file_group, stretch=3)
        top_layout.addWidget(control_group, stretch=1)
        main_layout.addLayout(top_layout)

        # --- 3. 状态标签 ---
        self.lbl_status = QLabel("状态: 准备就绪")
        main_layout.addWidget(self.lbl_status)

        # --- 4. 底部可拖拽区域 (结果 和 日志) ---
        # [修改] 移除了顶部的词典 QSplitter
        
        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 4a. 处理结果表格
        results_group = QGroupBox("处理结果")
        results_layout = QVBoxLayout()
        self.tbl_results = QTableWidget()
        self.tbl_results.setColumnCount(5) 
        self.tbl_results.setHorizontalHeaderLabels(["序号", "开始时间", "结束时间", "说话人", "翻译内容 (双击查看)"])
        self.tbl_results.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tbl_results.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.tbl_results.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tbl_results.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        results_layout.addWidget(self.tbl_results)
        results_group.setLayout(results_layout)
        
        # 4b. 运行日志
        log_group = QGroupBox("运行日志")
        log_layout = QVBoxLayout()
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        log_layout.addWidget(self.log_display)
        log_group.setLayout(log_layout)
        
        bottom_splitter.addWidget(results_group)
        bottom_splitter.addWidget(log_group)
        bottom_splitter.setSizes([1000, 500]) # 初始比例

        main_layout.addWidget(bottom_splitter, stretch=1) # [修改] 直接添加到主布局
        
        # --- 5. 设置主窗口 ---
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        
        # --- 6. 设置日志重定向 ---
        self.log_handler = QtLogHandler()
        self.log_handler.new_log_message.connect(self.update_log_display)
        if logger.handlers:
             self.log_handler.setFormatter(logger.handlers[0].formatter)
        logger.addHandler(self.log_handler)

    def browse_audio_file(self):
        """
        打开文件浏览器选择 .wav 文件
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择音频文件",
            str(settings.INPUT_WAV_FOLDER), # 默认打开 data/input/wav
            "音频文件 (*.wav)"
        )
        if file_path:
            self.input_file_path = Path(file_path)
            self.le_audio_file.setText(file_path)
            logger.info(f"已选择文件: {file_path}")

    # --- [删除] 词典管理函数 (add_glossary_row, del_glossary_row) ---

    def run_process(self):
        """
        点击“开始处理”按钮时调用的函数
        """
        if not self.input_file_path:
            logger.warning("请先选择一个音频文件！")
            self.lbl_status.setText("状态: 错误！请先选择文件")
            return
            
        if self.agent_thread and self.agent_thread.isRunning():
            logger.warning("一个任务正在处理中，请稍候...")
            return

        self.log_display.clear()
        self.tbl_results.setRowCount(0) # 清空表格
        logger.info("========= GUI 请求启动处理流程 =========")
        
        # --- [新功能] 读取动漫名称 ---
        anime_name = self.le_anime_name.text().strip()
        if anime_name:
            logger.info(f"读取指定动漫: {anime_name}")
        else:
            logger.info("未指定动漫，将自动检测。")
        # --- 结束 ---
        
        # [修改] 创建带路径和动漫名称的线程
        self.agent_thread = AgentThread(self.input_file_path, anime_name)
        self.agent_thread.start()
        
        self.btn_start.setEnabled(False)
        self.btn_start.setText("正在处理中...")
        self.lbl_status.setText("状态: 正在处理中...")
        
        self.agent_thread.finished_with_result.connect(self.process_finished)

    def process_finished(self, output_json_path):
        """
        当 AgentThread.run() 结束后调用的函数
        """
        logger.info("========= GUI 检测到流程已结束 =========")
        self.btn_start.setEnabled(True)
        self.btn_start.setText("开始处理")
        
        if output_json_path and output_json_path != 'None':
            self.lbl_status.setText(f"状态: 处理完毕！结果已保存。")
            self.populate_table(Path(output_json_path))
        else:
            self.lbl_status.setText("状态: 处理失败！请查看日志。")

    def populate_table(self, json_path):
        """
        读取 JSON 结果并填充表格
        """
        logger.info(f"正在从 {json_path.name} 加载结果到表格...")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.tbl_results.setRowCount(len(data))
            
            for i, item in enumerate(data):
                self.tbl_results.setItem(i, 0, QTableWidgetItem(str(i + 1)))
                self.tbl_results.setItem(i, 1, QTableWidgetItem(item.get("start_time", "")))
                self.tbl_results.setItem(i, 2, QTableWidgetItem(item.get("end_time", "")))
                self.tbl_results.setItem(i, 3, QTableWidgetItem(item.get("speaker_id", "N/A")))
                
                dialogue = item.get("dialogue_content", "")
                dialogue_item = QTableWidgetItem(dialogue)
                dialogue_item.setToolTip(dialogue) 
                self.tbl_results.setItem(i, 4, dialogue_item)

            self.tbl_results.resizeRowsToContents()
            logger.info(f"表格加载完毕，共 {len(data)} 条记录。")

        except Exception as e:
            logger.error(f"填充结果表格失败: {e}", exc_info=True)

    def open_output_dir(self):
        """
        打开最终的 JSON 输出文件夹
        """
        output_dir = settings.FINAL_OUTPUT_JSON_FOLDER
        output_dir.mkdir(parents=True, exist_ok=True) 
        
        url = QUrl.fromLocalFile(str(output_dir.resolve()))
        logger.info(f"正在打开输出目录: {url.toLocalFile()}")
        QDesktopServices.openUrl(url)

    def clear_log(self):
        """
        清空日志显示框
        """
        self.log_display.clear()

    def update_log_display(self, message):
        """
        接收来自 logger 的新消息并将其附加到文本框
        """
        self.log_display.append(message)
        self.log_display.verticalScrollBar().setValue(
            self.log_display.verticalScrollBar().maximum()
        )

    def closeEvent(self, event):
        """
        重写“关闭窗口”事件，确保后台线程也一起退出
        """
        logger.info("正在关闭应用程序...")
        if self.agent_thread and self.agent_thread.isRunning():
            self.agent_thread.quit()
            self.agent_thread.wait()
        event.accept()