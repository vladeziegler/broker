from crewai import Crew
from crewai import Process
from agents import Agents
from tasks import Tasks
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from transformers import pipeline
from langchain_openai import ChatOpenAI


load_dotenv()

#groq_api_key = os.getenv('GROQ_API_KEY')
#Groq = ChatGroq(
#    api_key=groq_api_key,
#    model="mixtral-8x7b-32768"
#)


api_key=os.getenv("OPENAI_API_KEY")
Groq = ChatOpenAI(
    api_key=api_key,
    model_name="gpt-4o", temperature=0.4,
    max_tokens=800)

def create_crew(company, job_id, url):
    agents = Agents()
    tasks = Tasks()

    set_up_agent = agents.set_up(company, url)
    set_up_quality_assurance_agent = agents.set_up_quality_assurance_agent(company, url)
    industry_analyst_agent = agents.industry_analyst(company, url)
    economic_analyst_agent = agents.economist(company, url)
    customer_research_agent = agents.customer_research(company, url)
    financial_analyst_agent = agents.financial_analyst(company, url)
    senior_quality_assurance_agent = agents.senior_quality_assurance(company, url)

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

    industry_research_task = tasks.industry_research(
        industry_analyst_agent,
        company,
        url,
        [set_up_research_task, quality_setup_assurance_review_task]
    )

    economic_research_task = tasks.economic_research(
        economic_analyst_agent,
        company,
        url,
        [set_up_research_task, quality_setup_assurance_review_task]
    )
    customer_research_task = tasks.customer_research(
        customer_research_agent,
        company,
        url,
        [set_up_research_task, quality_setup_assurance_review_task]
    )

    financial_research_task = tasks.financial_research(
        financial_analyst_agent,
        company,
        url,
        [set_up_research_task, quality_setup_assurance_review_task]
    )
    
    crew = Crew(
        agents=[set_up_agent, set_up_quality_assurance_agent, industry_analyst_agent, economic_analyst_agent, customer_research_agent, financial_analyst_agent, senior_quality_assurance_agent],
        tasks=[quality_setup_assurance_review_task, set_up_research_task, industry_research_task, economic_research_task, customer_research_task, financial_research_task],
        verbose=True,
        max_rpm=29,
        memory=False,
        manager_llm=Groq,
        process=Process.hierarchical,
#        embedder={
#            "provider": "mistralai",
#            "config": {
#                "model": "mistral-embed",
#                }}
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
