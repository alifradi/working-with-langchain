from langchain_huggingface import HuggingFaceEndpoint
import os
# Set your Hugging Face API token locally and dont share it
huggingfacehub_api_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# Define the LLM
# Define the LLM with the task
llm = HuggingFaceEndpoint(repo_id='tiiuae/falcon-7b-instruct', task='text-generation', huggingfacehub_api_token=huggingfacehub_api_token)
# Predict the words following the text in question
question = 'Whatever you do, take care of your shoes'
output = llm.predict(question)

print(output)

# exerecise 2

from langchain_core.prompts import PromptTemplate

# Create a prompt template from the template string
template = "You are an artificial intelligence assistant, answer the question. {question}"
prompt = PromptTemplate(template=template, input_variables=["question"])

# Create a chain to integrate the prompt template and LLM
llm = HuggingFaceEndpoint(repo_id='tiiuae/falcon-7b-instruct', task='text-generation', huggingfacehub_api_token=huggingfacehub_api_token)
llm_chain = prompt | llm

question = "How does LangChain make LLM application development easier?"
print(llm_chain.invoke({"question": question}))


# exercise 3
from langchain.prompts import PromptTemplate

# Create a prompt template from the template string
template = "You are an artificial intelligence assistant, answer the question. {question}"
prompt = PromptTemplate(template=template, input_variables=["question"])

# Create a chain to integrate the prompt template and LLM
llm = HuggingFaceEndpoint(repo_id='tiiuae/falcon-7b-instruct', task='text-generation', huggingfacehub_api_token=huggingfacehub_api_token)
llm_chain = prompt | llm

question = "How does LangChain make LLM application development easier?"
print(llm_chain.invoke({"question": question}))

# exercice 4

# Define an OpenAI chat model
llm = ChatOpenAI(model="gpt-4o-mini", api_key='<OPENAI_API_TOKEN>')		

# Create a chat prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Respond to question: {question}")
    ]
)

# Chain the prompt template and model, and invoke the chain
llm_chain = prompt_template | llm
response = llm_chain.invoke({"question": "How can I retain learning?"})
print(response.content)

# exercice 5

from langchain.prompts import PromptTemplate, FewShotPromptTemplate

# Create the examples list of dicts
examples = [
  {
    "question": "How many DataCamp courses has Jack completed?",
    "answer": "36"
  },
  {
    "question": "How much XP does Jack have on DataCamp?",
    "answer": "284,320XP"
  },
  {
    "question": "What technology does Jack learn about most on DataCamp?",
    "answer": "Python"
  }
]

# Complete the prompt for formatting answers
example_prompt = PromptTemplate.from_template("Question: {question}\n{answer}")

# Create the few-shot prompt
prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question: {input}",
    input_variables=["input"],
)

# Invoke the prompt template
prompt = prompt_template.invoke({"input": "What is Jack's favorite technology on DataCamp?"})
print(prompt.text)


# exercise 6
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# Create a prompt template that takes an input activity
learning_prompt = PromptTemplate(
    input_variables=["activity"],
    template="I want to learn how to {activity}. Can you suggest how I can learn this step-by-step?"
)

# Create a prompt template that places a time constraint on the output
time_prompt = PromptTemplate(
    input_variables=["learning_plan"],
    template="I only have one week. Can you create a plan to help me hit this goal: {learning_plan}."
)

# Invoke the learning_prompt with an activity
print(learning_prompt.invoke({"activity": "play golf"}))


# Complete the sequential chain with LCEL
seq_chain = ({"learning_plan": learning_prompt | llm | StrOutputParser()}
    | time_prompt
    | llm
    | StrOutputParser())

# Call the chain
print(seq_chain.invoke({"activity": "play the harmonica"}))


# exercise 7
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from langchain_community.agent_toolkits.load_tools import load_tools
# Define the tools
tools = load_tools(["wikipedia"])

# Define the agent
agent = create_react_agent(llm, tools)

# Invoke the agent
response = agent.invoke({"messages": [("human", "How many people live in New York City?")]})
print(response['messages'][-1].content)

# exercise 8

@tool
def retrieve_customer_info(name: str) -> str:
    """Retrieve customer information based on their name."""
    customer_info = customers[customers['name'] == name]
    return customer_info.to_string()

# Create a ReAct agent
agent = create_react_agent(llm, [retrieve_customer_info])

print(create_react_agent.__module__)

# Invoke the agent on the input
messages = agent.invoke({"messages": [("human", "Create a summary of our customer: Peak Performance Co.")]})
print(messages['messages'][-1].content)