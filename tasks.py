# tasks.py
from crewai import Task
from agents import strategist, researcher, synthesizer # Import your agents

# --- TASK DEFINITIONS ---

# Task 1: Create the Research Plan (Unchanged)
task_plan = Task(
    description=(
        "1. Analyze the user's high-level research goal: '{goal}'.\n"
        "2. Create a step-by-step research plan. This plan should be a "
        "list of 3-5 specific, targeted search queries for the researcher to execute.\n"
        "3. Your final output MUST be JUST the list of queries, "
        "one query per line. Do not add any other text."
    ),
    expected_output="A newline-separated list of 3-5 specific search queries.",
    agent=strategist,
)

# --- THE FIX IS HERE ---
# Task 2: Execute the Research (New, more flexible description)
task_research = Task(
    description=(
        "Research the user's original goal: '{goal}'.\n"
        "Use the list of 5 search queries provided by the strategist (available in your context) as your guide.\n"
        "You MUST find information that addresses all the key topics in the strategist's plan.\n"
        "Perform multiple searches using your 'Advanced Web Search' tool to gather all necessary information.\n"
        "Collect all the results (content and source) from all your searches.\n"
        "Your final output MUST be a single string containing all the "
        "collated research findings, clearly separated."
    ),
    expected_output="A single string containing all formatted research results, including content and sources.",
    agent=researcher,
    context=[task_plan], # This task depends on the output of task_plan
)
# -------------------------

# Task 3: Write the Final Report (Unchanged)
task_report = Task(
    description=(
        "1. Take the collated research findings from the researcher.\n"
        "2. Analyze the user's original goal: '{goal}'.\n"
        "3. Write a comprehensive, professional report that fully answers the user's goal.\n"
        "4. The report must be well-structured, with an introduction, "
        "bulleted key findings, and a conclusion.\n"
        "5. **CRITICAL:** You MUST cite your sources for the information. "
        "Format citations like [Source: http://example.com]."
    ),
    expected_output="A comprehensive, well-structured report with citations.",
    agent=synthesizer,
    context=[task_research], # This task depends on the output of task_research
)