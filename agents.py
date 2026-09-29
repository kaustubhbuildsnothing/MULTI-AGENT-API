import os
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

# LLM used   
llm = ChatOpenAI(model="gpt-4o")

def run_research_crew(topic: str) -> str:
    # 1. agents ka role aur goal define kara hai niche
    researcher = Agent(
        role="Senior Technology Researcher",
        goal=f"Uncover cutting-edge developments about {topic}",
        backstory="You are an expert tech analyst known for uncovering the most accurate and up-to-date information in the tech industry.",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    writer = Agent(
        role="Tech Content Strategist",
        goal=f"Craft a compelling, easy-to-understand blog post about {topic} based on the researcher's findings.",
        backstory="You are a renowned tech writer who simplifies complex topics into engaging, accessible content.",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    # 2. task defining.
    research_task = Task(
        description=f"Conduct a comprehensive analysis on the topic: {topic}. Identify key trends and major players.",
        expected_output="A detailed bulleted report summarizing all core aspects and trends.",
        agent=researcher
    )

    writing_task = Task(
        description=f"Using the insights from the researcher's report, write a 3-paragraph engaging blog post about {topic}.",
        expected_output="A 3-paragraph blog post formatted in markdown.",
        agent=writer
    )

    #crew workflow yaha se hai
    tech_crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        process=Process.sequential # Executes strictly in order: Research -> Write
    )


    result = tech_crew.kickoff()
    return str(result)