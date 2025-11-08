# Autonomous Market Research Agent 🤖

This project is a multi-agent AI system built with **CrewAI** that autonomously performs complex, multi-step research and generates comprehensive, cited reports from a single, high-level goal.

It's a full-stack application with a **Streamlit** web interface, powered by **Groq** (for high-speed LLM inference) and a custom "Pro-RAG" tool using **Tavily** and **Cohere Re-rank**.

---



## Core Features

* **Autonomous Multi-Agent System:** Built with CrewAI, the system uses a "crew" of specialized agents:
    * **Chief Research Strategist:** Analyzes the user's high-level goal and creates a detailed, step-by-step research plan.
    * **Expert Web Researcher:** Executes each step of the plan, using a custom "Pro-RAG" tool to find the most relevant information.
    * **Lead Report Synthesizer:** Takes the collated research from the Researcher and writes the final, professional, and fully-cited report.

* **"Pro-RAG" Search Tool:** The Researcher agent doesn't use a simple search. It uses a custom-built `advanced_search` tool that:
    1.  **Transforms Queries:** Uses an LLM to rewrite a simple user query into multiple, high-quality search queries.
    2.  **Retrieves:** Gathers documents for all transformed queries using the Tavily search API.
    3.  **Re-Ranks:** Uses the Cohere Re-ranker to find the single most relevant fact from the entire batch, dramatically reducing noise and improving accuracy.

* **Robust Engineering:**
    * **Proactive Rate-Limiting:** The application includes a custom `litellm` callback to add a 2-second delay between all LLM calls, preventing API rate-limit crashes.
    * **Agent Caching:** The Researcher agent's memory is set to `cache=True` to summarize its findings. This keeps the context window small, efficient, and well below token limits for complex, multi-step tasks.

* **Full-Stack Interface:** A simple and clean web UI built with Streamlit allows any non-technical user to run the entire autonomous research crew.

---

## 🛠️ Tech Stack

* **Orchestration:** CrewAI
* **LLM:** Groq (via `llama-3.1-8b-instant`)
* **Web UI:** Streamlit
* **Search:** Tavily API
* **Re-ranking:** Cohere API
* **Core Libraries:** LangChain, LiteLLM
* **Language:** Python

---

## 🚀 How to Run

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/your-username/autonomous-market-research-agent.git](https://github.com/your-username/autonomous-market-research-agent.git)
    cd autonomous-market-research-agent
    ```

2.  **Create and Activate a Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create Your `.env` File**
    Create a file named `.env` in the root of the project folder and add your API keys:
    ```env
    # .env file
    GROQ_API_KEY="your_groq_api_key_here"
    TAVILY_API_KEY="your_tavily_api_key_here"
    COHERE_API_KEY="your_cohere_api_key_here"
    ```

5.  **Run the Streamlit App!**
    ```bash
    streamlit run web_app.py
    ```
    Your browser will automatically open to the app.

