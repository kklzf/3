from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone #向量数据库
import os
from keys import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_API_ENV
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Pinecone
import streamlit as st #网站创建
import gtts #文字转语音



# App framework
# 如何创建自己的网页机器人
st.title('😷奈艾斯AI牙医🩺🦷') #用streamlit app创建一个标题
# 创建一个输入栏可以让用户去输入问题
query = st.text_input('欢迎来到AI牙科诊所,你可以问我关于牙科的问题，例如：洗一次牙多少钱？')

my_bar = st.progress(0, text='等待投喂问题哦')
# initialize search
# 开始搜索，解答
if query:
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_API_ENV
    )
    # llm是用来定义语言模型，在下面的例子，用的是openai，注意，此openai调用的是langchain方法不是openai本ai
    llm = OpenAI(temperature=0, max_tokens=-1, openai_api_key=OPENAI_API_KEY)
    print('1:'+ str(llm))
    my_bar.progress(10, text='正在查询新华字典')
    # embedding就是把文字变成数字
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    print('2:'+ str(embeddings))
    # 调用pinecone的数据库，开始查询任务
    docsearch = Pinecone.from_existing_index('dental',embedding=embeddings)
    print('3:'+ str(docsearch))
    # 相似度搜索，例如疼678，痛679，搜索用户的问题的相似度
    docs = docsearch.similarity_search(query, k=3)
    print('4:'+ str(docs))
    my_bar.progress(60, text='找到点头绪了')
    # 调用langchain的load qa办法，’stuff‘为一种放入openai的办法
    chain = load_qa_chain(llm, chain_type='stuff', verbose=True)
    print('5:'+ str(chain))
    my_bar.progress(90, text='可以开始生成答案了，脑细胞在燃烧')
    # 得到答案
    answer = chain.run(input_documents=docs, question=query, verbose=True)
    print('6:'+ str(answer))
    my_bar.progress(100, text='好了')
    st.write(answer)
    audio = gtts.gTTS(answer, lang='zh')
    audio.save("audio.wav")
    st.audio('audio.wav', start_time=0)
    os.remove("audio.wav")
