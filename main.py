from dotenv import load_dotenv
import os
from langchain_community.llms import OpenAI
import streamlit as st
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory




load_dotenv()


KEY = os.getenv("OPEN_API_KEY")



st.title("Langchain Demo with OPENAI Key")
input_text = st.text_input("Enter the any topic you want to search")


# Memory

person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
descr_memory = ConversationBufferMemory(input_key='dob', memory_key='description_history')

first_input_prompt = PromptTemplate(
    input_variables = ['name'],
    template = "tell me about about celebrity {name}"
)

llm = OpenAI(temperature=0.8,openai_api_key=KEY)

chain = LLMChain(llm=llm,prompt = first_input_prompt,output_key = 'person',verbose=True,memory=person_memory)


second_input_prompt = PromptTemplate(
    input_variables = ['person'],
    template = "when was {person} born ?"
)


chain2= LLMChain(llm=llm,prompt = second_input_prompt,output_key = 'dob',verbose=True,memory=dob_memory)


third_input_prompt=PromptTemplate(
    input_variables=['dob'],
    template="Mention 5 major events happened around {dob} in the world"
)

chain3=LLMChain(llm=llm,prompt=third_input_prompt,verbose=True,output_key='description',memory=descr_memory)



parent_chain = SequentialChain(chains = [chain,chain2,chain3],input_variables = ['name'],output_variables=['person','dob','description'],verbose=True)


if input_text:
    st.write(parent_chain.invoke(input_text))
    with st.expander('Person Name'): 
        st.info(person_memory.buffer)

    with st.expander('Major Events'): 
        st.info(descr_memory.buffer)



