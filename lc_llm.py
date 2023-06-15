from __future__ import annotations

import logging
import sys
import warnings
from typing import (
    AbstractSet,
    Any,
    Callable,
    Collection,
    Dict,
    Generator,
    List,
    Literal,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)
from langchain.llms.base import LLM
from langchain.schema import Generation, LLMResult
from langchain.utils import get_from_dict_or_env
import requests

class HostedOpenAIChat(LLM):
    server_url:str = "http://localhost:8000"
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.0
    uid: str = None
    @property
    def _llm_type(self) -> str:
        return "hosted_openai_chat"

    def _call(self, prompt:str, stop: Optional[List[str]] = None) -> str:
        headers = {"Content-Type": "application/json"}
        response = requests.post(
        self.server_url+"/api/completion/",
        json={
            "prompt": [{"role": "user", "content": prompt}],
            "model_name": self.model,
            "token":self.uid,
            "temperature": self.temperature
            
        },
        headers=headers,
        )
        try:
            json_data = response.json()
            return json_data["choices"][0]["message"]["content"]
        except:
            print(response.content)
    
