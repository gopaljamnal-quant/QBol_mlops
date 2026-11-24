import os
from dotenv import load_dotenv

from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
)

response = client.responses.create(
    model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    input="What is the result of USA mid term elections?",
)

print(f"answer: {response.output_text}")