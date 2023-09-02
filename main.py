from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone #向量数据库
import os
from keys import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_API_ENV
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Pinecone
import streamlit as st #网站创建
import gtts #文字转语音

#'''
#下面的这部分代码是将文件夹中的word文档，上传到自己的向量数据库
#'''
#首先进入文件夹查看数据
directory_path = 'dental_data' #这边填入你自己的数据文件所在的文件夹
data = []
# loop through each file in the directory
for filename in os.listdir(directory_path):
    # check if the file is a doc or docx file
    # 检查所有doc以及docx后缀的文件
    if filename.endswith(".doc") or filename.endswith(".docx"):
        # print the file name
        # langchain自带功能，加载word文档
        loader = UnstructuredWordDocumentLoader(f'{directory_path}/{filename}')
        print(loader)
        data.append(loader.load())
print(len(data))
#Chunking the data into smaller pieces
#再用菜刀把文档分隔开，chunk_size就是我们要切多大，建议设置700及以下，因为openai有字数限制，chunk_overlap就是重复上下文多少个字
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
texts = []
for i in range(len(data)):
    print(i)
    texts.append(text_splitter.split_documents(data[i]))
    print(text_splitter.split_documents(data[i]))
print(len(texts))

#Creating embeddings
# 把文字转成数字
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# initialize pinecone
# 把数字放进向量数据库，environment填写你的数据库所在的位置，例如useast
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_API_ENV
)
# 要填入对应的index name
index_name = "dental-index" # put in the name of your pinecone index here
for i in range(len(texts)):
    Pinecone.from_texts([t.page_content for t in texts[i]], embeddings, index_name=index_name)
    print("done")


'''
存储完成向量数据库之后，我们就可以运行下面的代码，用streamlit帮我们做一个简单的网页可以用来调用我们的机器人问答
'''
# App framework
# 如何创建自己的网页机器人
st.title('奈艾斯AI牙医🩺🦷') #用streamlit app创建一个标题
# 创建一个输入栏可以让用户去输入问题
query = st.text_input('欢迎来到奈艾斯AI牙科诊所,你可以问我关于牙科的问题，例如：洗一次牙多少钱？')

my_bar = st.progress(0, text='等待投喂问题哦')
# initialize search
# 开始搜索，解答
if query:
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_API_ENV
    )
    #llm是用来定义语言模型，在下面的例子，用的是openai，注意，此openai调用的是langchain方法不是openai本ai
    llm = OpenAI(temperature=0, max_tokens=-1, openai_api_key=OPENAI_API_KEY)
    print('1:'+ str(llm))
    my_bar.progress(10, text='正在查询奈艾斯AI智库')
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
    my_bar.progress(100, text='好了ikun')
    st.write(answer)
    audio = gtts.gTTS(answer, lang='zh')
    audio.save("audio.wav")
    st.audio('audio.wav', start_time=0)
    os.remove("audio.wav")

# '''
# 以下代码删除所有indexes模块，默认注释关闭，谨慎使用
# '''
# Are you sure you want to delete this? If yes, delete this line of code
# # deleting all indexes
# pinecone.init(
#     api_key=PINECONE_API_KEY,
#     environment=PINECONE_API_ENV
# )
# pinecone.Index('dental').delete(delete_all=True)
