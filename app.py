# app.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file FIRST.
load_dotenv()

# Now, import all the other libraries
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_cohere import CohereRerank

# --- FIX (FINAL) ---
# The correct import path is in 'langchain_classic'
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
# -----------

# Import TavilySearchAPIRetriever from langchain_community
from langchain_community.retrievers.tavily_search_api import TavilySearchAPIRetriever


# --- Part A: Query Transformation (with Groq) ---

# 1. Initialize your LLM (Groq)
# Updated to use a supported model - llama3-8b-8192 has been decommissioned
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# 2. Create a "Query Transformation" prompt
query_transform_prompt = ChatPromptTemplate.from_template(
    """You are an expert at rewriting a user's query into 3 distinct, high-quality search queries
    that will get the best possible results from a web search engine (Tavily).
    
    Original Query: {query}
    
    Return ONLY a newline-separated list of the 3 new queries. Do not add any preamble.
    """
)

# 3. Create the transformation "chain"
query_transform_chain = query_transform_prompt | llm | StrOutputParser()

# 4. Test it!
original_query = "latest on Llama 3"
print(f"--- Original Query: '{original_query}' ---")
generated_queries = query_transform_chain.invoke({"query": original_query})

print("--- Generated Queries ---")
print(generated_queries)


# --- Part B: Retrieve & Re-Rank ---

# 1. Split the generated queries into a list
query_list = generated_queries.split("\n")

# 2. Retrieve ALL documents for ALL queries
all_docs = []
for q in query_list:
    if q.strip(): # Ensure query is not empty
        print(f"--- Searching for: {q.strip()} ---")
        retriever = TavilySearchAPIRetriever(k=5)
        all_docs.extend(retriever.invoke(q.strip()))

print(f"\n--- Total Docs Retrieved: {len(all_docs)} ---")

# 3. Set up the Re-Ranker
cohere_reranker = CohereRerank(model="rerank-english-v3.0", top_n=3)

# 4. Create a simple retriever that returns our documents
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List

class SimpleRetriever(BaseRetriever):
    documents: List[Document]

    def _get_relevant_documents(self, query: str) -> List[Document]:
        return self.documents

simple_retriever = SimpleRetriever(documents=all_docs)

# 5. Create the Compression Retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=cohere_reranker,
    base_retriever=simple_retriever
)

# 6. Run the final Re-Ranking!
print(f"\n--- Running Re-Ranker on '{original_query}' ---")
reranked_results = compression_retriever.invoke(original_query)

# 6. Display the final, high-quality results
print("\n--- FINAL RE-RANKED RESULTS ---")
for i, res in enumerate(reranked_results):
    print(f"RESULT {i+1}:")
    print(f"{res.page_content}")
    print(f"  (Source: {res.metadata['source']})\n")