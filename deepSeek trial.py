# use Hugging Face endpoint to try deepseek llm for text generation

from langchain_huggingface import HuggingFaceEndpoint
import os
# Set your Hugging Face API token locally and dont share it
huggingfacehub_api_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
llm = HuggingFaceEndpoint(repo_id='deepseek-ai/DeepSeek-R1', task='text-generation', huggingfacehub_api_token=huggingfacehub_api_token)
question = 'Generate a story about tunisia'
output = llm.predict(question)
print(output)