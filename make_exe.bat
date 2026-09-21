python make_build_time.py
pyinstaller --clean --onefile --add-binary "ffmpeg.exe;." --add-binary "ffprobe.exe;."  nextwave_ui.py
copy dist\nextwave_ui.exe .