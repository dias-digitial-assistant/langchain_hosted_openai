from __future__ import annotations
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Set,
    Tuple,
    Union,
    Optional,
)
import numpy as np
from pydantic import BaseModel
from langchain.embeddings.base import Embeddings
import  requests
import tiktoken
from tenacity import retry, stop_after_attempt, wait_fixed


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

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def _embed(self, texts: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        # First check if we have a list of texts or a single text
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            self.server_url + "/api/embedding/",
            json={
                "input": texts,
                "model_name": "text-embedding-ada-002",
                "token": self.uid
            },
            headers=headers,
        )
        try:
            if response.status_code == 200:
                json_data = response.json()
                if isinstance(texts, str):
                    return json_data['data'][0]['embedding']
                else:
                    embeddings = [json_data['data'][i]['embedding'] for i in range(len(texts))]
                    return embeddings
            else:
                #throw an exception
                raise Exception("Error in embedding")
        except:
            if response.status_code != 200:
                print(response.content)
            else:
                print("Data formatting incorrect")
    
    def _get_len_safe_embeddings(
    self, texts: List[str], *, chunk_size: Optional[int] = None
) -> List[List[float]]:
        embeddings: List[List[float]] = [[] for _ in range(len(texts))]
        try:
            import tiktoken

            tokens = []
            indices = []
            encoding = tiktoken.model.encoding_for_model(self.model)
            for i, text in enumerate(texts):
                # replace newlines, which can negatively affect performance.
                text = text.replace("\n", " ")
                token = encoding.encode(text)
                for j in range(0, len(token), self.embedding_ctx_length):
                    tokens.append(token[j : j + self.embedding_ctx_length])
                    indices.append(i)

            batched_embeddings = []
            _chunk_size = chunk_size or self.chunk_size
            for i in range(0, len(tokens), _chunk_size):
                decoded_tokens = [encoding.decode(t) for t in tokens[i : i + _chunk_size]]
                response = self._embed(
                    texts=decoded_tokens,
                )
                batched_embeddings += [r for r in response]

            results: List[List[List[float]]] = [[] for _ in range(len(texts))]
            lens: List[List[int]] = [[] for _ in range(len(texts))]
            for i in range(len(indices)):
                results[indices[i]].append(batched_embeddings[i])
                lens[indices[i]].append(len(batched_embeddings[i]))

            for i in range(len(texts)):
                average = np.average(results[i], axis=0, weights=lens[i])
                embeddings[i] = (average / np.linalg.norm(average)).tolist()

            return embeddings

        except ImportError:
            raise ValueError(
                "Could not import tiktoken python package. "
                "This is needed in order to for OpenAIEmbeddings. "
                "Please install it with `pip install tiktoken`."
            )



    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to Hosted OpenAIs Document endpoint.
        Args:
            texts: The list of texts to embed.
        Returns:
            List of embeddings, one for each text.
        """
        return self._get_len_safe_embeddings(texts)
        

    def embed_query(self, text: str) -> List[float]:
        """Call out to  Hosted OpenAIs, query embedding endpoint
        Args:
            text: The text to embed.
        Returns:
            Embeddings for the text.
        """
        embs = self._get_len_safe_embeddings([text])
        return embs[0]
    
