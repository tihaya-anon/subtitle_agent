# 文件: main.py (根目录)

import sys
import os

# 确保 src 目录在 Python 路径上
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# from PySide6.QtWidgets import QApplication
# from src.logger import logger


# def main():
#     """
#     项目主入口
#     """
#     logger.info("正在启动 GUI 应用程序...")
#     try:
#         # 1. 创建 QApplication 实例
#         app = QApplication(sys.argv)

#         # 2. 导入并显示主窗口
#         # 注意：我们在函数内部导入，以确保 logger 和 sys.path 优先设置
#         from src.gui.main_window import MainWindow

#         window = MainWindow()
#         window.show()

#         # 3. 运行 Qt 事件循环
#         sys.exit(app.exec())

#     except ImportError as e:
#         logger.critical(f"启动失败：缺少必要的 GUI 模块 (PySide6)。错误: {e}")
#         logger.critical("请确保你已经运行了: pip install -r requirements.txt")
#         sys.exit(1)
#     except Exception as e:
#         logger.critical(f"GUI 发生未捕获的致命错误: {e}", exc_info=True)
#         sys.exit(1)


def cli():
    from src.agent import run_single_file_process
    from pathlib import Path

    run_single_file_process(Path("./test.wav"), "BanG Dream! It's MyGO!!!!!")


if __name__ == "__main__":
    # main()
    cli()
