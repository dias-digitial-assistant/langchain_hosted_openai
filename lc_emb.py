from __future__ import annotations
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Set,
    Tuple,
    Union,
)
import numpy as np
from pydantic import BaseModel
from langchain.embeddings.base import Embeddings
import  requests

class HostedOpenAIEmbeddings(BaseModel, Embeddings):
    uid: str = None
    server_url:str = "http://localhost:8000"
    client: Any  #: :meta private:
    model: str = "text-embedding-ada-002"
    embedding_ctx_length: int = 8191
    allowed_special: Union[Literal["all"], Set[str]] = set()
    disallowed_special: Union[Literal["all"], Set[str], Tuple[()]] = "all"
    chunk_size: int = 1000
    """Maximum number of texts to embed in each batch"""
    max_retries: int = 6
    """Maximum number of retries to make when generating."""


    def _embed(self, texts: Union[List[str],str]):
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            self.server_url+"/api/embedding/",
            json={
                "input": texts,
                "model_name": "text-embedding-ada-002",
                "token":self.uid
            },
            headers=headers,
        )
        try:
            if response.status_code == 200:
                json_data = response.json()
                if isinstance(texts, str):                    
                    return json_data['data'][0]['embedding']
                else:
                    embeddings = [json_data['data'][i]['embedding']for i in range(len(texts))]
                    return embeddings
        except:
            if response.status_code != 200:
                print(response.content)
            else:
                print("Data formatting incorrect")


    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to Hosted OpenAIs Document endpoint.
        Args:
            texts: The list of texts to embed.
        Returns:
            List of embeddings, one for each text.
        """
        return self._embed(texts)
        

    def embed_query(self, text: str) -> List[float]:
        """Call out to  Hosted OpenAIs, query embedding endpoint
        Args:
            text: The text to embed.
        Returns:
            Embeddings for the text.
        """
        return self._embed(text)