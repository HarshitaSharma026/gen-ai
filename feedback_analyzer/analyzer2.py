from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
import streamlit as st

load_dotenv()

# two models, one for each chain running in parallel
model1 = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
model2 = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

output_parser = StrOutputParser()

# defining schema for sentiment response 
class Feedback(BaseModel):
  sentiment: str = Field(description="Classify the sentiment and return output in json format")

# instantiating pydantic parser
pydantic_parser = PydanticOutputParser(pydantic_object=Feedback)

# prompt to get the sentiment from user question
prompt_sentiment = PromptTemplate(
  template="Classify the sentiment of the following feedback text into positive or negative. \n {feedback} \n {format_instructions}",
  input_variables=['feedback'],
  partial_variables={'format_instructions': pydantic_parser.get_format_instructions()}
)
# prompt to generate summary
prompt_summarize = PromptTemplate(
  template="Generate a short summary of the given user feedback. \n {feedback}",
  input_variables=['feedback']
)

# parallel chain, getting sentiment and summary parallely.
parallel_chain = RunnableParallel({
  'sentiment': prompt_sentiment | model1 | pydantic_parser,
  'summary': prompt_summarize | model2 | output_parser
})

# get sentiment function, it converts the "Feedback" pydantic object into python dictionary and sends the value (positive or negative) of sentiment further. 
def get_sentiment(sentimentObj):
  parsed = sentimentObj['sentiment'].model_dump()
  return parsed['sentiment']


# create branching and new prompts for positive and negative sentiments
prompt_positive = PromptTemplate(
  template="Identify the key strengths mentioned in the summary. Structure your answer by first mentioning the key strengths and then suggesting 3 ways to enhance them even more.\n {summary}",
  input_variables=['summary']
)
prompt_negative = PromptTemplate(
  template="Identify the key problems mentioned in the summary. Structure your answer by first mentioning the key problems and then suggesting 3 fast to implement ways to remove such problems.\n {summary}",
  input_variables=['summary']
)

# branching chain to define the branch and what needs to be done at each branch
branching_chain = RunnableBranch(
  (lambda x: get_sentiment(x) == 'positive', prompt_positive | model1 | output_parser),
  (lambda x: get_sentiment(x) == 'negative', prompt_negative | model2 | output_parser),
  RunnableLambda(lambda x: "Mixed emotions, not able to find proper sentiment.")
)

# combining parallel and branching chain
final_chain = parallel_chain | branching_chain

# streamlit code - for UI
st.header('Feedback analyzer')
st.title('Understand employee feedbacks better.')
user_input = st.text_area('Type your feedback here: ')

if st.button('Analyze'):
  result = final_chain.invoke({'feedback': user_input})
  st.write(result)


# feedback1 : The team here is very helpful. i have learnt a lot here. i have found good friends, and collegues here. 
# feedback 2: Being a fresher last year, i really enjoyed the project and the learning i got. however as we progressed further in the project i was able to see micro-management within some people. After our manager has been changed, i can see people having no skills getting promotion and people having real skills are getting lagged behind. attending team meetings started feeling suffocated.
