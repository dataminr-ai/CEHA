from typing import List, Optional, Any, Dict
from abc import abstractmethod
import requests
import json
from textwrap import dedent
from openai import OpenAI
from mistralai import Mistral


class LLMCaller:
    @abstractmethod
    def __call__(
        self,
        prompt,
        adapter: Optional[str] = None,
        sampling_params: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        pass


class OpenAILLMCaller(LLMCaller):
    def __init__(self, openai_api_key, model_name, header=None, timeout=60):
        """
        Initialise a LLM
        :param dic header: header for the request
        :param int timeout: Timeout for the request to LLM service
        """
        self.client = OpenAI(api_key=openai_api_key)
        self.timeout = timeout
        self.model_name = model_name
    
    def __call__(self, prompt, sampling_params={}) -> List[str]:
        response =  self.client.chat.completions.create(
                model=self.model_name,
                temperature=0.001,
                top_p=0.001,
            messages=prompt
        )
        return [response.choices[0].message.content.strip()]
    
    
class MistralAiLLMCaller(LLMCaller):
    def __init__(self, mistralai_api_key, model_name, header=None, timeout=60):
        """
        Initialise a LLM
        :param str model_route: model_route of the desired model
        :param dic header: header for the request
        :param int timeout: Timeout for the request to LLM service
        """
        self.client = Mistral(api_key=mistralai_api_key)
        self.timeout = timeout
        self.model_name = model_name
    
    def __call__(self, prompt, sampling_params={}) -> List[str]:
        response =  self.client.chat.complete(
                model=self.model_name,
                messages=prompt
        )
        return [response.choices[0].message.content.strip()] 
