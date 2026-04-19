from pydantic import BaseModel, Field
from typing import List, Optional
from crewai import Agent, Task, Crew
from crewai.project import CrewBase, agent, task, crew

class TicketClassification(BaseModel):
    intent: str = Field(..., description="billing, technical, feature, bug")
    urgency: str = Field(..., description="low, medium, high")
    sentiment: str = Field(..., description="positive, neutral, negative")

class KnowledgeResult(BaseModel):
    sources: List[str]
    summary: str

class DraftResponse(BaseModel):
    message: str
    confidence: float = Field(..., ge=0, le=1)

class QAResult(BaseModel):
    approved: bool
    issues: Optional[List[str]] = []



@CrewBase
class SaaSSupportCrew:

    # -------- AGENTS -------- #

    @agent
    def triage_agent(self) -> Agent:
        return Agent(
            role="Triage Specialist",
            goal="Classify incoming support tickets with intent, urgency, and sentiment",
            backstory="Expert in customer support triaging for SaaS platforms",
            verbose=True
        )

    @agent
    def knowledge_agent(self) -> Agent:
        return Agent(
            role="Knowledge Retriever",
            goal="Retrieve relevant documentation and past solutions",
            backstory="Expert in searching internal knowledge bases and docs",
            verbose=True
        )

    @agent
    def resolution_agent(self) -> Agent:
        return Agent(
            role="Support Responder",
            goal="Generate helpful, accurate, and empathetic responses",
            backstory="Experienced SaaS support engineer",
            verbose=True
        )

    @agent
    def qa_agent(self) -> Agent:
        return Agent(
            role="QA & Compliance Reviewer",
            goal="Ensure responses are accurate, safe, and well-toned",
            backstory="Strict reviewer ensuring no hallucinations or bad tone",
            verbose=True
        )

    # -------- TASKS -------- #

    @task
    def classify_ticket(self) -> Task:
        return Task(
            description="""
            Analyze the incoming support ticket and classify:
            - intent (billing, technical, feature, bug)
            - urgency (low, medium, high)
            - sentiment (positive, neutral, negative)
            """,
            expected_output="Structured classification of the ticket",
            agent=self.triage_agent(),
            output_json=TicketClassification
        )

    @task
    def retrieve_knowledge(self) -> Task:
        return Task(
            description="""
            Based on the classified ticket, retrieve relevant knowledge:
            - Help docs
            - API references
            - Past resolved tickets
            """,
            expected_output="Relevant sources and summarized knowledge",
            agent=self.knowledge_agent(),
            output_json=KnowledgeResult
        )

    @task
    def generate_response(self) -> Task:
        return Task(
            description="""
            Generate a helpful response using:
            - ticket classification
            - retrieved knowledge

            Ensure:
            - Clear steps
            - Empathetic tone
            - Accuracy
            """,
            expected_output="Draft response with confidence score",
            agent=self.resolution_agent(),
            output_json=DraftResponse
        )

    @task
    def qa_review(self) -> Task:
        return Task(
            description="""
            Review the drafted response:
            - Check for hallucinations
            - Ensure correctness
            - Validate tone

            Approve or reject with issues.
            """,
            expected_output="QA approval result",
            agent=self.qa_agent(),
            output_json=QAResult
        )

    # -------- CREW -------- #

    @crew
    def support_crew(self) -> Crew:
        return Crew(
            agents=[
                self.triage_agent(),
                self.knowledge_agent(),
                self.resolution_agent(),
                self.qa_agent()
            ],
            tasks=[
                self.classify_ticket(),
                self.retrieve_knowledge(),
                self.generate_response(),
                self.qa_review()
            ],
            verbose=True
        )