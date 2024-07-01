from crewai import Crew
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from transformers import pipeline

# Agents file------------------------------------------------------------------------------------------------------------
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

api_key = os.getenv('OPENAI_API_KEY')
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
      self.llm = ChatGroq(
          api_key=groq_api_key,
          model="mixtral-8x7b-32768"
      )
    
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
       tools=[company_tool, scraping_tool],
        verbose=True,
        llm=self.llm,
        max_iter=3) # Add closing parenthesis here

  def set_up_quality_assurance_agent(self,company, url):
    return Agent(
        role='Senior Setup Quality Assurance Specialist',
        goal=f"""Get recognition for your work and ensure that the information provided by the Senior Company Analyst about {company} is accurate, sharp, timely, and clear.""",
        backstory= f"""You need to make sure the Senior Company Analyst makes no assumptions and stops once he's considered that his content is valuable enough.""",
        allow_delegation=True,
        verbose=True,
        llm=self.llm,
        max_iter=3)
  

  # Tasks file------------------------------------------------------------------------------------------------------------

from crewai import Task
from textwrap import dedent
from datetime import date
import os
from langchain_openai import ChatOpenAI

class Tasks():

  def set_up_research(self, agent, company, url):
    return Task(description=f"""
        Investigate what {company} is all about. This means that you
        describe its activities, but also assess its overall position in its industry and value chain.
        You make sure to use find all the necessary information to describe the company, its role in the value chain, the profile of its customers, its target audience, product and services range.
        It has to be an comprehensive and accurate report that encompases all the key aspects of the company.
        {self.__tip_section()}""",
      expected_output=f"""A concise report about {company}. The report should include:
      **1.** {company} description.
      **2.** The key problems {company} is trying to solve with its products and services
      **3.** The description and benefits of each of its product range and services
      **4** An overview of its product range and categories
      **5** An overview of its customer segments based on the customer information, testimonials, but also given the product range
      **6** The industry and sub-industry {company} is in, and the main competitors
      **7** The position of the {company} in the value chain of its industry and its split between B2B and B2C segments
      **8** The regions the company is operating
      **9** The main distribution channels, and the main marketing channels it is using
      **10** The existing relationships with financial institutions such as banks, investors, etc.
      **11** The main customer segments, and the main customer needs, their applications, and use cases
      **12** The {company} positioning based on pricing, messaging and product range
      The aim of this report is to provide everything that the next agents will need to understand {company} and its industry later on.
      You stop once you've considered that your content is valuable enough.""",
#      async_execution=True,
      agent=agent,
      output_file=f"{os.getcwd()}/outputs/{date.today()}_set_up_research_{company}.txt")


  def quality_setup_assurance_review(self, agent, company, url):
    return Task(description=f"""
        Review the response drafted by the Senior Company Analyst about {company}. Ensure it is comprehensive, relevant, accurate, and actionable.
        Make sure to verify the information provided and ensure that no assumptions are made. Provide feedback to the senior company analyst on the
        quality of the response, ensuring it will contain all the key information to understand {company}.
        {self.__tip_section()}""",
#        async_execution=True,
        expected_output=f"""A final, concise, timely, and actionable reports that defines {company}.""",
        agent=agent)

  def __tip_section(self):
    return "If you do your BEST WORK, I'll tip you $10000! and you will get promoted to the next level!"
  
  # Main file ------------------------------------------------------------------------------------------------------------

groq_api_key = os.getenv('GROQ_API_KEY')
Groq = ChatGroq(
    api_key=groq_api_key,
    model="mixtral-8x7b-32768"
)

def create_crew(company, job_id, url):
    agents = Agents()
    tasks = Tasks()

    set_up_agent = agents.set_up(company, url)
    set_up_quality_assurance_agent = agents.set_up_quality_assurance_agent(company, url)

    set_up_research_task = tasks.set_up_research(
        set_up_agent,
        company,
        url
    )
    quality_setup_assurance_review_task = tasks.quality_setup_assurance_review(
        set_up_quality_assurance_agent,
        company,
        url
    )
    
    crew = Crew(
        agents=[set_up_agent, set_up_quality_assurance_agent],
        tasks=[quality_setup_assurance_review_task],
        verbose=True,
        max_rpm=29,
        memory=False,
        manager_llm=Groq,
        embedder={
            "provider": "mistralai",
            "config": {
                "model": "mistral-embed",
                }}
            )

    return crew

class MyCrew:
    def __init__(self, company, job_id, url):
        self.company = company
        self.job_id = job_id
        self.url = url
        self.crew = create_crew(company, job_id, url)

    def run(self):
        result = self.crew.kickoff(inputs={'company': self.company, 'url': self.url})
        return result

if __name__ == "__main__":
    # Prompt the user for input
    company = input("Please enter the name of the company you want to analyze: ")
    url = input("Please enter the URL of the company you want to analyze: ")
    
    # Create and run the crew with the user's input
    my_crew = MyCrew(company=company, job_id="user_input_job", url=url)
    result = my_crew.run()
    print(result)
