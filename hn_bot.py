
import os
import re
import math
import json

from openai import OpenAI
from react import ReactAgent
from tools import get_hn_stories, get_relevant_comments, get_story_content


def get_hn_bot():
  bot_system_prompt = """You are the Singularity Incarnation of Hacker News.
  The human will ask you for information about Hacker News.
  If you can't find any information  about the question asked
  or the result is incomplete, apologise to the human and ask him if
  you can help him with something else.
  If the human asks you to show him stories, do it using a markdown table.
  The markdown table has the following format:

  story_id | title | url | score"""

  agent = ReactAgent(
      system_prompt=bot_system_prompt,
      tools=[get_hn_stories, get_relevant_comments, get_story_content]
  )
  return agent
