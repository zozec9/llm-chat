import pandas as pd
import numpy as np
import streamlit as st
import time
# streamlit run app.py

# 部署在streamlit云端

st.title("hello world")
st.write("hello 陶姐你好")
st.info("hello 娜娜好")
st.header("好")


# 聊天机器人页面，可视化展示内容和大模型的聊天记录。
from pydantic import BaseModel

# 构建和大模型的聊天chain
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import AIMessage,HumanMessage

# 定义大模型
model = ChatTongyi(
    model_name="qwen-max",
    dashscope_api_key="sk-2eeffb28e0184545b8d2bb7d238c9a75",
    streaming=True
)

# 从哪里去获取聊天内容
memory_key = 'history'
# 定义提示词模板
prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name=memory_key),
        ('human', '{input}')
    ]
)

class Message(BaseModel):
    content: str
    role: str

if "messages" not in st.session_state:
    st.session_state.messages = []

def to_message_place_holder(messages):
    return [
        AIMessage(content=message['content']) if message['role'] == 'ai'
        else HumanMessage(content=message['content'])
        for message in messages
    ]

chain = {
    'input': lambda x: x['input'],
    'history': lambda x: to_message_place_holder(x['messages'])
} | prompt | model | StrOutputParser()


# 页面左半部分展示聊天内容，右半部分展示聊天记录

left,right = st.columns([0.7, 0.3])

with left:

    # 聊天内容展示
    container = st.container()
    with container:
        for message in st.session_state.messages:
            with st.chat_message(message['role']):
                st.write(message['content'])

    # 接收用户输入，存放在session_state中
    prompt = st.chat_input("您好，请问有什么可以帮助您的吗？")
    if prompt:
        st.session_state.messages.append(Message(content=prompt, role='human').model_dump())
        with container:
            with st.chat_message("human"):
                st.write(prompt)

        # 获取大模型的返回，并展示
        with container:
            response = st.write_stream(chain.stream({'input': prompt, 'messages':st.session_state.messages }))
        st.session_state.messages.append(Message(content=response, role='ai').model_dump())
with right:
    # 聊天记录展示
    st.json(st.session_state.messages)
