from lc_llm import HostedOpenAIChat
from dotenv import load_dotenv
load_dotenv()
import os
if __name__=="__main__":
    llm = HostedOpenAIChat(uid=os.getenv("DEMO_CLIENT_UID"), server_url=os.getenv("SERVER_URL"), model="gpt-4")
    output = llm("Du bist der Assitant. Was ist 2+2?")
    print(output)
