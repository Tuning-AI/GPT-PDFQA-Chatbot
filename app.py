import streamlit as st
from PIL import Image
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
os.environ["OPENAI_API_KEY"] = "YOUR OPENAI_API_KEY"
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.question_answering import load_qa_chain
from elevenlabs import generate, play
from elevenlabs import set_api_key

set_api_key("YOUR elevenlabs api_key ")
def audiof(text) : 
    audio = generate(text=text,voice="Charlie")
    play(audio=audio)
img = Image.open("PDF.png")
st.set_page_config(page_title='pdfQA Chatbot',page_icon = img)
st.sidebar.image("PDF.png" , width=80)
temperature = st.sidebar.slider("Select your temperature value : " ,min_value=0.1 , max_value=1.0 , value=0.9)
use_audio = st.sidebar.checkbox("Use audio output")
st.header(":hand: Welcome To pdfQA Chatbot : ")
st.info("The PDFQA Chatbot is an intelligent tool designed to answer questions related to PDF documents. With advanced natural language processing capabilities, it offers accurate and efficient responses, enhancing document understanding and accessibility.")
pdffile = st.file_uploader("Please upload your pdf file to start The conversation: " , type=['pdf'])
if pdffile : 
    # location of the pdf file/files. 
    doc_reader = PdfReader(pdffile.name)
    # read data from the file and put them into a variable called raw_text
    raw_text = ''
    for i, page in enumerate(doc_reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text
    # Splitting up the text into smaller chunks for indexing
    text_splitter = CharacterTextSplitter(        
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap  = 200, #striding over the text
        length_function = len,
    )
    texts = text_splitter.split_text(raw_text)
    # Download embeddings from OpenAI
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_texts(texts, embeddings , persist_directory="DB")
    llm = OpenAI(streaming=True,
             callbacks=[StreamingStdOutCallbackHandler()])
    chain = load_qa_chain(OpenAI(streaming=True,
                                 temperature=temperature,
                                 callbacks=[StreamingStdOutCallbackHandler()]), 
                                 chain_type="stuff")
    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "Hi :hand: my name is pdfQA chatbot, You can ask any thing about you pdf file content."}]

    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        docs = docsearch.similarity_search(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chain.run(input_documents=docs, question=prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        if use_audio : 
            audiof(full_response)
        st.session_state.messages.append(message)
