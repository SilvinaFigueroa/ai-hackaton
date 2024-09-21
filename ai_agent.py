# OpenAI Agent with LlamaIndex

from os import environ
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Set your OpenAI API key
environ["OPENAI_API_KEY"] = "sk-svcacct-JtMLhV4TL44uR-6S8w2vsx6l4lSLeU_VEJvMnjZ-SLpp1UlP5Cc1gej7-TBDvQLT3BlbkFJVOwTraN-qNmkpY3Gk4Y-rLeCW5avRpejNTYAkCvOSTjwGk4hUQNeFThYwa9NkAA"



# Import necessary components

# from llama_index.core import SimpleDirectoryReader, ServiceContext, VectorStoreIndex

from llama_index.readers.web import SimpleWebPageReader


from llama_index.core import (
    Document,
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
    load_index_from_storage,
)

from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI


# Create an LLM object to use for the QueryEngine and the ReActAgent
llm = OpenAI(model="gpt-4")

# Set up Phoenix
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

import phoenix as px
session = px.launch_app()

endpoint = "http://127.0.0.1:6006/v1/traces"  # Phoenix receiver address

tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

# Define your resume and job description as text variables
resume_text = """Silvina From Argentina to Brazil, Ireland, and now the United States, my journey has been defined by continuous learning and embracing new challenges. 
I’m an enthusiastic problem solver and love helping people, whether I’m managing projects, supporting a team, or building applications from scratch. I love seeing how technology can improve people’s lives, and that's why I'm committed to continuously learning and growing in this field.
My experience at Digital Aid Seattle has allowed me to apply my technical skills in real-world scenarios, utilizing tools like Airtable and Softr to streamline processes and create impactful solutions. I approach every project with a proactive mindset, focusing on empowering my team and driving collective success.
I’m actively seeking opportunities where I can combine my project management expertise with my technical skills to create meaningful solutions. If you share my passion for making a difference, let’s connect!."""

job_description_text = """The Operation Specialist/Trust Associate is responsible for providing administrative support to the Trust Services team. This includes processing trust transactions, preparing reports, and providing customer service. The ideal candidate will have a strong understanding of trust principles and procedures, as well as excellent customer service skills."""

# Create documents from the text

# Wrap the text in Document objects
resume_doc = Document(text=resume_text)
job_doc = Document(text=job_description_text)

# Build vector indexes from the documents
resume_index = VectorStoreIndex.from_documents([resume_doc])
job_index = VectorStoreIndex.from_documents([job_doc])

# Persist the indexes (optional if you want to save them for later)
resume_index.storage_context.persist(persist_dir="./storage/resume")
job_index.storage_context.persist(persist_dir="./storage/job_description")

# Set up query engines for resume and job description
resume_engine = resume_index.as_query_engine(similarity_top_k=3, llm=llm)
job_engine = job_index.as_query_engine(similarity_top_k=3, llm=llm)

# Define the query engines as tools to be used by the agent
query_engine_tools = [
    QueryEngineTool(
        query_engine=resume_engine,
        metadata=ToolMetadata(
            name="resume_engine",
            description="Provides information from the user's resume."
        ),
    ),
    QueryEngineTool(
        query_engine=job_engine,
        metadata=ToolMetadata(
            name="job_engine",
            description="Provides information from the job description."
        ),
    ),
]

# Instructions on how to craft a cover letter
cover_letter_template = """
Create a cover letter using the information from the resume and job description. Follow this instructions for the teamplate:
Cover Letter Template Instructions:

Greeting: Start with a formal greeting. If a hiring manager's name is known, use it; otherwise, "Dear Hiring Manager" is appropriate.

Introduction:

Introduce yourself briefly.
Mention the position you are applying for and the company name.
Explain why you are excited about the role and how it aligns with your career goals or aspirations.
First Paragraph (Experience and Skills):

Highlight key relevant experiences and skills that match the job description.
Emphasize technical and non-technical skills (e.g., problem-solving, project management).
Mention specific projects or achievements that demonstrate your abilities.
Second Paragraph (Relevance to Company):

Show enthusiasm for the company's mission or industry, even if it's new to you.
Describe how your background can be valuable to the company.
Mention your willingness to learn quickly and adapt to new environments if applicable.
Conclusion:

Reiterate your interest in the position.
Express your eagerness to contribute to the company's goals and mission.
Mention your availability for further discussions or interviews.
Closing: Use a polite closing statement such as "Thank you for considering my application."

Sign off with "Sincerely" or "Best regards."
Include your name and any relevant contact information (e.g., LinkedIn profile or personal website).

"""

# Create the agent with instructions
agent = ReActAgent.from_tools(
    query_engine_tools,
    llm=llm,
    verbose=True,
    max_turns=10,
)

# Ask the agent to craft the cover letter
response = agent.chat(cover_letter_template)
print(str(response))
