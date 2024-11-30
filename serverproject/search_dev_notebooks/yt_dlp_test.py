# DIRECTORY SET
import os
import sys
from pathlib import Path
base_dir=Path(os.getcwd()).parent
# os.chdir(os.path.join(base_dir, 'serverproject'))
os.chdir(base_dir)
print(os.getcwd())

# Load dotenv
import dotenv
dotenv.load_dotenv()

# DJANGO SETUP
import django
sys.path.append(os.path.abspath(''))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "serverproject.settings")
django.setup()

# Import async modules
import asyncio
from asgiref.sync import sync_to_async

# Import display modules
from IPython.display import display, Markdown

# Import other modules
import faiss
import time
import numpy as np




from destinyapp.models import StreamRecapData, FastRecapData

from core import services
from core import utils
from core import controller

import subprocess

video_id="1zu41rrc_Ng"
folder_path="destinyapp/working_folder/test_transcripts/"
if not os.path.isdir(folder_path):
    os.makedirs(folder_path)
output_path=folder_path+str(video_id)+'_transcript'

cookies_path="cookies.txt"
command=['yt-dlp', '--skip-download', '--write-auto-sub', '--sub-lang','en','-o', output_path, '--verbose', '--cookies', cookies_path, '--convert-subs', 'srt', 'https://www.youtube.com/watch?v='+video_id]
print(" ".join(command))
result = subprocess.run(command, check=True, capture_output=True, text=True)
print(result)
