from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from streamlit_chat import message
from utils import *
import os
import yaml

from langchain.agents import create_json_agent, AgentExecutor
from langchain.agents.agent_toolkits import JsonToolkit
from langchain.chains import LLMChain
from langchain.llms.openai import OpenAI
from langchain.requests import TextRequestsWrapper
from langchain.tools.json.tool import JsonSpec
openai.api_key = "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"
model = SentenceTransformer('all-MiniLM-L6-v2')
os.environ["OPENAI_API_KEY"] = "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"
#####################################################json
with open("openai_openapi.yml") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
json_spec = JsonSpec(dict_=data, max_value_length=4000)
json_toolkit = JsonToolkit(spec=json_spec)

json_agent_executor = create_json_agent(
    llm=OpenAI(temperature=0), toolkit=json_toolkit, verbose=True
)

######################################################



pinecone.init(api_key='e6fe16b5-86c0-461d-8efe-5911c598122e', environment='gcp-starter')
index = pinecone.Index('chatbot')


st.subheader("Chatbot ProfeSearch \n Connecting Graduate Students with Professors")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key="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")

if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)


system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context, 
and if the answer is not contained within the text below, say 'I don't know'""")


human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)




# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()


with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("typing..."):
            conversation_string = get_conversation_string()
            # st.code(conversation_string)
            refined_query = query_refiner(conversation_string, query) # convert user query to a nice query
            st.subheader("Refined Query:")
            st.write(refined_query)
            print("refined" ,refined_query)

            # context = find_match(refined_query)
            input_em = model.encode(refined_query).tolist()
            result = index.query(input_em, top_k=8, includeMetadata=True)
            context = result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']
            context2 =result['matches'][2]['metadata']['text']+"\n"+result['matches'][3]['metadata']['text']
            context3 = result['matches'][4]['metadata']['text']+"\n"+result['matches'][5]['metadata']['text']
            context4 =result['matches'][6]['metadata']['text']+"\n"+result['matches'][7]['metadata']['text']

            print("result" , result)
            # print(context)  # sum of results
            response1 = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{refined_query}") #convert response "the problem"
            print("response1" , response1) 
            response2 = conversation.predict(input=f"Context:\n {context2} \n\n Query:\n{refined_query}")
            print("response2" , response2) 
            response3 = conversation.predict(input=f"Context:\n {context3} \n\n Query:\n{refined_query}")
            print("response3" , response3) 
            response4 = conversation.predict(input=f"Context:\n {context4} \n\n Query:\n{refined_query}")
            print("response4" , response4) 

            response5 = json_agent_executor.run(refined_query) # json
            print("response5" , response5) 

            # response = conversation.predict(input=f"Context:\n {response1+response2+response3} \n\n Query:\n{query}")
            prompt = f"""
                your task is helping a user to find appropriate professor . just anser based on provided Text.
                  do not add  anything other than provided text.

                Text:
                ```{response1+response2+response3 +response4 +response5}```

                user request:
                ```{refined_query}```
                """
            response = get_completion(prompt)

            # print("response" , response.choices[0].message["content"])  
        st.session_state.requests.append(query)
        st.session_state.responses.append(response.choices[0].message["content"]) 
        

with response_container:
    if st.session_state['responses']:

        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')

          
