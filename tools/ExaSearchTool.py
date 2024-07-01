from exa_py import Exa
from langchain.agents import tool
import os
from dotenv import load_dotenv

load_dotenv()

exa_key = os.getenv('EXA_API_KEY')


class ExaSearchTool:
    @tool
    def search(query: str):
        """Search for a webpage based on the query."""
        return ExaSearchTool._exa().search(f"{query}", use_autoprompt=True, num_results=3)

    @tool
    def find_similar(url: str):
        """Search for webpages similar to a given URL.
        The url passed in should be a URL returned from `search`.
        """
        return ExaSearchTool._exa().find_similar(url, num_results=3)
    
    @tool
    def search_and_contents(query: str):    
        """Search for a webpage and get contents based on the query.""" 
        return ExaSearchTool._exa().search_and_contents(query, use_autoprompt=True, num_results=3)  

    def tools():   
        return [ExaSearchTool.search, ExaSearchTool.find_similar, ExaSearchTool.search_and_contents]  

    def _exa():   
        return Exa(api_key=exa_key)