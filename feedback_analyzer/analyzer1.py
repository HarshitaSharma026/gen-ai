from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain.schema.runnable import RunnableParallel
from pydantic import BaseModel, Field
import streamlit as st

load_dotenv()

# two models, one for each chain running in parallel
model1 = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
model2 = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

output_parser = StrOutputParser()

# defining schema for sentiment response 
class Feedback(BaseModel):
  sentiment: str = Field(description="Give sentiment of the given feedback as either positive or negative")

pydantic_parser = PydanticOutputParser(pydantic_object=Feedback)

# prompt for finding the sentiment of feedback
prompt_sentiment = PromptTemplate(
  template="Given the user feedback, extract the sentiment out of it.\n {feedback} \n {format_instructions}",
  input_variables=['feedback'],
  partial_variables={'format_instructions': pydantic_parser.get_format_instructions()}
)
# prompt for getting the summary
prompt_summarize = PromptTemplate(
  template="Generate a short summary of the given user feedback. \n {feedback}",
  input_variables=['feedback']
)
# prompt for suggestions
prompt_improve = PromptTemplate(
  template="Generate suggestions for improvement based on the given summary of the user feedback. \n {summary}",
  input_variables=['summary']
)

# implementing parallel chain to get sentiment and suggestions
parallel_chain = RunnableParallel({
  'sentiment': prompt_sentiment | model1 | output_parser,
  'suggestions': prompt_summarize | model2 | output_parser | prompt_improve | model2 | output_parser,
})

# final prompt to combine sentiment and suggestions and propose actionable advices.
prompt_combine = PromptTemplate(
  template="Take the sentiment and suggestions for improvement and merge them into a single document. The document should include the main problem the employee is facing and accordingly provide the 3 most effective suggestions that can be implemented as early as possible in the final document.Do not make the document too large. Keep it concise and to the point.\n Sentiment: {sentiment} and Suggestions: {suggestions}",
  input_variables=['sentiment', 'suggestions']
)

# implementing merge chain to merge the suggestions and sentiments and get a single document answer
merger_chain = prompt_combine | model1 |  output_parser

# final chain merging the above two chains
chain = parallel_chain | merger_chain

# streamlit code - for UI
st.header('Feedback analyzer')
st.title('Understand employee feedbacks better. (v1)')
user_input = st.text_area('Type your feedback here: ')

if st.button('Analyze'):
  result = chain.invoke({'feedback': user_input})
  st.write(result)


# feedback1 : The team here is very helpful. i have learnt a lot here. i have found good friends, and collegues here. 
# feedback 2: Being a fresher last year, i really enjoyed the project and the learning i got. however as we progressed further in the project i was able to see micro-management within some people. After our manager has been changed, i can see people having no skills getting promotion and people having real skills are getting lagged behind. attending team meetings started feeling suffocated.