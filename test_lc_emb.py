from lc_emb import HostedOpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
import os
if __name__=="__main__":
    embeddings = HostedOpenAIEmbeddings(uid=os.getenv("DEMO_CLIENT_UID"), server_url= os.getenv("SERVER_URL"))
    text = "This is a test document"
    query_result = embeddings.embed_query(text)
    doc_result = embeddings.embed_documents([text, "This is also a document"])

    print(len(query_result))
    print(len(doc_result))

