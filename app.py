
import os
import asyncio

from PIL import Image
import streamlit as st

from hn_bot import get_hn_bot
from dotenv import load_dotenv

load_dotenv()

# Set Streamlit page config
st.set_page_config(page_title="Sebastian's Bot 🤖📰")
st.title("Sebastian's Bot 🤖📰")


# Sidebar - API Key input
with st.sidebar:
    st.markdown("""
    # **Greetings, Digital Explorer!**

    Are you fatigued from navigating the expansive digital realm in search of your daily tech tales
    and hacker happenings? Fear not, for your cyber-savvy companion has descended upon the scene –
    behold the extraordinary **Sebastian's Bot**!
    """)

    st.session_state["agent"] = get_hn_bot()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []


def generate_response(question):
    """Generate response while passing conversation history as context."""
    context = "\n".join([msg['bot'] for msg in st.session_state["messages"]])
    response = st.session_state["agent"].run(f"Context: {context} Question: {question}")
    return response

# Display chat history
for msg in st.session_state["messages"]:
    st.chat_message("human").write(msg["user"])
    st.chat_message("ai").write(msg["bot"])

# Chat input handling
if prompt := st.chat_input():
    st.chat_message("human").write(prompt)
    with st.spinner("Thinking ..."):
        response = generate_response(prompt)
        st.chat_message("ai").write(response)

    # Store conversation in history
    st.session_state["messages"].append({"user": prompt, "bot": response})
