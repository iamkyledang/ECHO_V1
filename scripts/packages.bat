@echo off
call conda activate myenv
pip install spacy
python -m spacy download en_core_web_sm
pip install pyqt5
pip install pyside6
pip install ffmpeg
pip install openai
pip install moviepy
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install openai-whisper
pip install pyannote.audio
pip install pydub
pip install googletrans
pip install language_tool_python
conda install montreal-forced-aligner -y
