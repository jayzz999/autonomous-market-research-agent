# main.py
import litellm
from crewai import Crew, Process
from agents import strategist, researcher, synthesizer
from tasks import task_plan, task_research, task_report
import os
import time
from dotenv import load_dotenv

# Load .env file first
load_dotenv()

# --- COMPREHENSIVE RATE LIMIT FIX ---
litellm.num_retries = 15  
litellm.request_timeout = 600  

def custom_callback(kwargs, completion_response, start_time, end_time):
    """Add delay after each LLM call to avoid rate limits"""
    time.sleep(2)  # 2 second delay between all LLM calls

litellm.success_callback = [custom_callback]
os.environ["LITELLM_RETRY_STRATEGY"] = "exponential_backoff_retry"
# -----------------------------------------------------------

# --- DEFINE YOUR RESEARCH GOAL ---
research_goal = "Give me a competitive analysis of Tesla vs. Rivian, focusing on 2025 market sentiment, battery technology, and production numbers."

# --- ASSEMBLE THE CREW ---
crew = Crew(
    agents=[strategist, researcher, synthesizer],
    tasks=[task_plan, task_research, task_report],
    process=Process.sequential, 
    verbose=False
)

# --- This will now only run when you execute `python main.py` ---
if __name__ == "__main__":
    print("🚀 CrewAI Market Research Crew is ready. Kicking off...")
    print(f"Goal: {research_goal}")

    # This is where the magic happens
    report = crew.kickoff(inputs={'goal': research_goal})

    # --- PRINT THE FINAL REPORT ---
    print("\n\n--- 📜 FINAL REPORT ---")
    print(report)