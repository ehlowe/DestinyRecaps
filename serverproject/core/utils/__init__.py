import os
import faiss
import openai
from openai import AsyncOpenAI
import anthropic
from anthropic import AsyncAnthropic


import assemblyai as aai
openai_client=openai.OpenAI(api_key=os.environ.get("openai",""))
async_openai_client=AsyncOpenAI(api_key=os.environ.get("openai",""))
anthropic_client=anthropic.Anthropic(api_key=os.environ.get("anthropic",""))
async_anthropic_client=AsyncAnthropic(api_key=os.environ.get("anthropic",""))
os.environ["REPLICATE_API_TOKEN"]=os.environ.get("replicate","")
aai.settings.api_key=os.environ.get("assemblyai","")

import groq
groq_client=groq.Groq(api_key=os.environ.get("GROQ_API_KEY"))
from groq import AsyncGroq
async_groq_client=AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))


# Api related utils
from . import api_requests
from .api_requests import ModelNameEnum, async_response_handler, generate_image


# Database related utils
from . import database_operations
from .database_operations import save_data, get_recap_data, get_all_recaps, get_all_recaps_fast, get_all_weekly_recaps, get_plain_transcript, delete_stream_recap_data
