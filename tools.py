# tools.py
import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_cohere import CohereRerank
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.retrievers.tavily_search_api import TavilySearchAPIRetriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List
from crewai.tools import tool  # <-- IMPORT THE DECORATOR

# Load environment variables from .env file
load_dotenv()

# This is your custom class from app.py
class SimpleRetriever(BaseRetriever):
    """A simple retriever that just returns the documents it's given."""
    documents: List[Document]

    def _get_relevant_documents(self, query: str) -> List[Document]:
        return self.documents

# --- THE FIX: USE THE @tool DECORATOR ---
@tool("Advanced Web Search")
def advanced_search(original_query: str) -> List[dict]:
    """
    Performs an advanced, multi-step search and re-ranking
    to get the most relevant, high-quality information.
    Input should be a string (the research query).
    """
    
    # 1. Initialize models
    # Using llama-3.3-70b-versatile which has higher rate limits (30K TPM vs 6K TPM)
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    cohere_reranker = CohereRerank(model="rerank-english-v3.0", top_n=1) # Get top 1

    # 2. Create query transformation chain
    query_transform_prompt = ChatPromptTemplate.from_template(
        """You are an expert at rewriting a user's query into 2 distinct, high-quality search queries
        that will get the best possible results from a web search engine (Tavily).

        Original Query: {query}

        Return ONLY a newline-separated list of the 2 new queries. Do not add any preamble.
        """
    )
    query_transform_chain = query_transform_prompt | llm | StrOutputParser()

    # 3. Get new queries with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            generated_queries = query_transform_chain.invoke({"query": original_query})
            break
        except Exception as e:
            if "rate_limit" in str(e).lower() and attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # Exponential backoff: 2s, 4s, 6s
                print(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                time.sleep(wait_time)
            else:
                raise

    query_list = generated_queries.split("\n")
    print(f"--- Generated Queries: {query_list}")

    # 4. Retrieve all documents
    tavily_retriever = TavilySearchAPIRetriever(k=1) # Get 1 doc

    all_docs = []
    for q in query_list:
        if q.strip():
            print(f"--- Searching for: {q.strip()} ---")
            all_docs.extend(tavily_retriever.invoke(q.strip()))
            # Add small delay between searches to avoid overwhelming the API
            time.sleep(0.5)
            
    print(f"--- Total Docs Retrieved: {len(all_docs)} ---")

    # 5. Create our custom retriever
    simple_retriever = SimpleRetriever(documents=all_docs)
    
    # 6. Create and run the re-ranking retriever
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=cohere_reranker,
        base_retriever=simple_retriever
    )
    
    reranked_docs = compression_retriever.invoke(original_query)
    
    # 7. Format the output
    final_results = []
    for res in reranked_docs:
        final_results.append({
            "content": res.page_content,
            "source": res.metadata['source']
        })
        
    return final_results

# --- Test your new function ---
if __name__ == "__main__":
    # This block runs ONLY when you execute 'python tools.py' directly
    print("--- Testing Advanced Search Tool ---")
    
    results = advanced_search("What is the market sentiment for Tesla vs Rivian in 2025?")
    
    print("\n--- ADVANCED SEARCH FUNCTION RESULTS ---")
    if results:
        for res in results:
            print(f"- {res['content']}\n  (Source: {res['source']})\n")
    else:
        print("No results found.")