# agents.py
import os
from dotenv import load_dotenv
from crewai import Agent, LLM
from typing import List, Dict

# --- IMPORT YOUR CUSTOM TOOL ---
from tools import advanced_search 

# --- LOAD API KEYS ---
load_dotenv()
# ---------------------------

# --- Define the Groq LLM explicitly using LLM class ---
# We are back to the 70B model that is smart enough to work correctly.
groq_llm = LLM(
    model="groq/llama-3.3-70b-versatile",  
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7,
    max_tokens=2048
)
# ---------------------------------------------

# --- AGENT DEFINITIONS ---

# 3. Chief Strategist Agent
strategist = Agent(
    role="Chief Research Strategist",
    goal="Create a comprehensive, step-by-step research plan to answer a complex user goal.",
    backstory=(
        "You are a master strategist, an expert in breaking down complex problems. "
        "You analyze a user's high-level goal and produce a perfect, "
        "logical, and efficient research plan for your team to execute. "
        "Your plan is the blueprint for success."
    ),
    llm=groq_llm,
    allow_delegation=False,
    verbose=False
)

# 4. Expert Researcher Agent
researcher = Agent(
    role="Expert Web Researcher",
    goal="Execute individual research tasks by finding the best information on the web.",
    backstory=(
        "You are a world-class researcher. You are given a specific research query "
        "and your job is to use your 'Advanced Web Search' tool to find the most "
        "relevant, high-quality, and factual information. You are meticulous and "
        "only return facts."
    ),
    llm=groq_llm,
    tools=[advanced_search],
    allow_delegation=False,
    verbose=False,
    
    # --- RE-ENABLE CACHING ---
    # This prevents token-limit errors on single runs
    cache=True 
    # -------------------------
)

# 5. Lead Report Synthesizer Agent
synthesizer = Agent(
    role="Lead Report Synthesizer",
    goal="Write a final, comprehensive, and well-structured report based on the research findings.",
    backstory=(
        "You are an expert editor and writer. You are given a collection of research snippets, "
        "facts, and sources. Your job is to synthesize this information into a "
        "flawless, professional, and easy-to-read report. You MUST only use the "
        "information provided and you MUST cite your sources."
    ),
    llm=groq_llm,
    allow_delegation=False,
    verbose=False
)