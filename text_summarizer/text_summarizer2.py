# 2nd version of summarizer that allows us to define the schema of json output we want

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

schema = [
  ResponseSchema(name="role", description="Extract the role from the user question. If no role is mentioned take 'Expert' as default."),
  ResponseSchema(name="topic", description="extract the topic to explain from the user question"),
  ResponseSchema(name="lines", description="Extract number of lines the user want the answer in. If nothing is given take 4 as default number of lines.")
]

parser = StructuredOutputParser.from_response_schemas(schema)


template1 = PromptTemplate(
  template="""Take the user input: {user_input} and extract the role you have to play, the topic to explain, and the number of lines.\n {format_instruction}""",
  input_variables=['user_input'],
  partial_variables={'format_instruction': parser.get_format_instructions()}
)

template2 = PromptTemplate(
  template="""You are a helpful {role}. Answer the user question : {user_question} in {lines} lines.""",
  input_variables=['role', 'user_question', 'lines']
)


st.header('Topic Summarizer')
# st.title('Write your query here: ')
user_input = st.text_input('Write your query here')

if st.button('Summarize'):
  output1 = template1 | model | parser
  inter_result = output1.invoke({'user_input': user_input})
  print(f"\n\n {inter_result}\n Type of line: {type(inter_result['lines'])} \n Check: {inter_result['lines'] * 20}")

  output2 = template2 | model | StrOutputParser()
  result = output2.invoke({
    'role': inter_result['role'],
    'user_question': inter_result['topic'],
    'lines': inter_result['lines']
  })

  st.write(result)



key_points = """
    - used StructureOutputParsers() for giving the schema structure.
    - default value instructions given to the model is working fine. 
"""

problem = """
    - 'lines' in inter_result is a string, it should be a number.
    - if i store it in another file, it'll be stored as a string.
    - no data validation.
    - if i try to multiply no of lines with 20, its actually taking it as string and printing it 20 times.
"""


