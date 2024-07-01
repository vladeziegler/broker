from crewai import Agent
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, WebsiteSearchTool
from langchain.tools import tool
from langchain.agents import load_tools
import os
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from cachetools import TTLCache, cached
from tools.ExaSearchTool import ExaSearchTool
from transformers import pipeline


load_dotenv()

#api_key = os.getenv('OPENAI_API_KEY')
api_key = "sk-proj-PME5zyuoC5f07xa7Y7wRT3BlbkFJaK0N5uGxLQNPaIzs0IBG"
os.environ["OPENAI_API_KEY"] = "sk-proj-PME5zyuoC5f07xa7Y7wRT3BlbkFJaK0N5uGxLQNPaIzs0IBG"
groq_api_key = os.getenv('GROQ_API_KEY')
exa_key = os.getenv('EXA_API_KEY')

cache = TTLCache(maxsize=100, ttl=3600)

search_tool = SerperDevTool()
#company_tool = WebsiteSearchTool()
exa_search_tool = ExaSearchTool()

scraping_tool = WebsiteSearchTool(
    config=dict(
        llm=dict(
            provider="mistralai",
            config=dict(
                model="mixtral-8x7b-32768",
                temperature=0.2,
            ),
        ),
        embedder=dict(
            provider="mistralai",
            config=dict(
              model="mistral-embed"
                                      ),
        ),
    )
)

class Agents():
  def __init__(self):
      self.llm = ChatOpenAI(
      api_key=os.getenv("OPENAI_API_KEY"),
          model_name="gpt-4o", temperature=0.4,
          max_tokens=800)

  #def __init__(self):
  #    self.llm = ChatGroq(
  #        api_key=groq_api_key,
  #        model="mixtral-8x7b-32768"
  #    )
    
  def set_up(self,company, url):
    company_tool = WebsiteSearchTool(url)
    return Agent(
        role='Senior Company Analyst',
        goal=f"""Establish a clear profile of the {company}.""",
        backstory= f""" You are an expert in understanding how companies work.
        This means you can describe what {company} key problems it is addressing, its benefits, and its target audience.
        Importantly, you can tell which industry, and sub-industry {company} is in, and what are the main competitors.
        Based on the products and services of {company}, you can tell what are the main use cases and the main features.
        But also, you can tell where {company} lies in the value chain of its industry. For instance, is it a supplier, a distributor, a retailer, or a manufacturer?
        You can also tell in which regions the company is operating.
        Based on the customers of {company}, you can tell what are the main customer segments, and what are the main customer needs, their applications, and use cases.
        You can also tell what are the main distribution channels, and the main marketing channels.
        Also, you can tell me if {company} has existing relationships with financial institutions such as banks, investors, etc.
        You make no assumptions and always verify the information before sharing it with the team.
        """,
       allow_delegation=False,
#       tools=[website_search_tool],
       tools=[company_tool],
        verbose=True,
#        llm=ollama3,
        llm=self.llm,
        max_iter=3,
#        max_time=600
    ) # Add closing parenthesis here

  def set_up_quality_assurance_agent(self,company, url):
    return Agent(
        role='Senior Setup Quality Assurance Specialist',
        goal=f"""Get recognition for your work and ensure that the information provided by the Senior Company Analyst about {company} is accurate, sharp, timely, and clear.""",
        backstory= f"""You need to make sure the Senior Company Analyst makes no assumptions and stops once he's considered that his content is valuable enough.""",
        allow_delegation=True,
        verbose=True,
      #  llm=ollama3,
        llm=self.llm,
        max_iter=3,
  #      max_time=600
    )
  
  def industry_analyst(self,company, url):
    return Agent(
        role='Senior Industry Analyst',
        goal=f"""Given the information provided by the Senior Company Analyst, provide a comprehensive analysis about the industry of {company}. Focus on the country, and region in which {company} is operating.""",
        backstory= f"""You have worked as a market research consultant for 20 years, generating industry reports for companies
         such as Euromonitor, Statista. You're extremely data-driven, and you leverage all the knowledge of the internet to identify the market dynamics, headwinds, tailwinds, opportunities, risks 
         of a given industry. You have an eye for global trends, but focus you remain hyper focused on the region and industry in which {company} is operating.""",
        allow_delegation=False,
        verbose=True,
        tools=[search_tool],
      #  llm=ollama3,
        llm=self.llm,
        max_iter=3
        )


  def economist(self,company, url):
    return Agent(
        role='Senior Economist',
        goal=f"""Given the information provided by the Senior Company Analyst, provide a comprehensive economic analysis about the unit econnomics in the industry of {company}. Remain focused on the country, and region in which {company} is operating.""",
        backstory= f"""You have worked as an economist for consultancies who seek to benchmark activities of companies across the industry. You've worked with companies such as McKinsey, BCG, and Deloitte. You're extremely data-driven, and you leverage all 
        the knowledge of the internet to identify the various economic indicators that are relevant to the industry in which {company} is operating.
        This means that you research metrics such as:
        - Gross margins
        - Net margins
        - Average revenue per user
        - Customer acquisition costs
        - Average order value
        - Main cost drivers
        - Main revenue drivers
         """,
        allow_delegation=False,
        verbose=True,
        tools=[search_tool],
      #  llm=ollama3,
        llm=self.llm,
        max_iter=3
        )

  def customer_research(self,company, url):
    return Agent(
        role='Senior Customer Research Analyst',
        goal=f"""Given the information provided by the Senior Company Analyst, provide a comprehensive analysis about the buyer profile, user persona and Ideal Customeer Profile (ICP) about the customers of {company} and its competitors. Focus only on the information you can find about its specific country, and region in which {company} is operating.""",
        backstory= f"""You have worked as a customer researcher and strategy analyst for consultancies who seek to understand demand dynamics in any given industry. Importantly, you want to assess the health and quality of these customers, but also their purchasing behaviour. You've worked with companies such as McKinsey, BCG, and Deloitte. You're extremely data-driven, and you leverage all 
        the knowledge of the internet to identify the information needed about the industry in which {company} is operating.
        This means that you will research topics such as:
        - The buyer profile of {company}
        - The user persona of {company}
        - The Ideal Customer Profile (ICP) of {company}
        - The decision criteria driving the purchase for {company} products and services
        - The buying behaviour of the customers of {company} and its competitors
        - The role of the customer in the value chain of the industry
        - The role of {company} on their customers cost structure and overall operations
        - The major events that could lead a customer to stop paying, churning, defaulting, or switching to another the company
         """,
        allow_delegation=False,
        verbose=True,
        tools=[search_tool],
      #  llm=ollama3,
        llm=self.llm,
        max_iter=3
        )
  

  def financial_analyst(self,company, url):
    return Agent(
        role='Senior Financial Analyst',
        goal=f"""Given the information provided by the Senior Company Analyst, provide a comprehensive analysis about the market dynamics about the industry and competitors of {company}.""",
        backstory= f"""You have worked as in Investment Banking, Private Equity, and consultancies to help with financial and market due diligences. You're excellent at identifying patterns in markets. You're excellent at identifying if strategic or financial investors are picking up interest in a given industry or the competitors of {company}. 
        You've worked with companies such as McKinsey, BCG, and Deloitte. You're extremely data-driven, and you leverage all 
        the knowledge of the internet to identify the information needed about the financial markets dynamics in which {company} is operating.
        This means that you will research topics such as:
        - If there were recent M&A, mergers in the industry of {company}. Focus only on its main country of operation. Mention the deals that have been made, and the companies involved.
        - You will find the names of the most active strategic acquirers and investors in the space of {company}.
        - You will compile a list of the relevant acquisition transactions that took place with companies similar to {company}. This means companies in the same industry, possibly region, and similar size.
        - You will identify the reasons that could make it interesting for investors and strategic acquirers to invest in this space. This can range from wanting to consolidate, to expanding into new regions, adding new products, tapping into new segments, etc.
         """,
        allow_delegation=False,
        verbose=True,
        tools=[search_tool],
      #  llm=ollama3,
        llm=self.llm,
        max_iter=3
        )

  def senior_quality_assurance(self,company, url):
    return Agent(
        role='Senior Quality Assurance Analyst',
        goal=f"""Review all the information provided by ALL the other agents regarding {company} ensuring their work meets their requirements and fit their expected output. The other agents include:
        - Senior Company Analyst
        - Senior Setup Quality Assurance Specialist
        - Senior Industry Analyst
        - Senior Economist
        - Senior Customer Research Analyst
        - Senior Financial Analyst
        """,
        backstory= f"""You are excellent at meeting client expectations. You've worked in Investment Banking, Private Equity, and consultancies and know exactly what information 
        is needed to make an informed decision about potentially investor in a company and industy. 
        You need to make sure all the information provided by the other agents is accurate, sharp, timely, and clear.""",
        allow_delegation=True,
        verbose=True,
        tools=[search_tool],
      #  llm=ollama3,
        llm=self.llm,
        max_iter=3
        )