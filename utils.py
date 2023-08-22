
from sentence_transformers import SentenceTransformer
import pinecone
import openai
import streamlit as st
openai.api_key = "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"
model = SentenceTransformer('all-MiniLM-L6-v2')

pinecone.init(api_key='e6fe16b5-86c0-461d-8efe-5911c598122e', environment='gcp-starter')
index = pinecone.Index('chatbot')

university = "toronto university"
def query_refiner(conversation, query):
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=f"""your task is helping a user to find appropriate some {university} professors . 
    Given the following user query and conversation log, formulate a question that would be the most relevant 
    to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:""",
    temperature=0,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response['choices'][0]['text']

def find_match(input):
    input_em = model.encode(input).tolist()
    # print("input_em" , input_em)
    result = index.query(input_em, top_k=4, includeMetadata=True)
    print("result" , result)
    # docs = index.similarity_search(input)
    # print("docs :",docs)
    return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']+"\n"+result['matches'][2]['metadata']['text']+"\n"+result['matches'][3]['metadata']['text'] 

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    # print(response)
    return response