# eval.py
import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

# --- Import your crew from main.py ---
from main import crew, research_goal
# Import your tasks to access their output
from tasks import task_research, task_report

# 1. DEFINE YOUR TEST SET
# These are the "questions" we will test your crew on.
test_questions = [
    research_goal, # The one you just ran
    "What is the future of autonomous driving? Compare Waymo, Tesla, and Cruise.",
    "Summarize the main arguments for and against generative AI in creative industries for 2025.",
    "Provide a market analysis of the top 3 cloud providers (AWS, Azure, GCP) for 2025, focusing on AI services."
]

# 2. RUN THE CREW AND COLLECT RESULTS
results = []
for question in test_questions:
    print(f"\n--- Evaluating Question: {question} ---")
    
    # Run the crew
    report = crew.kickoff(inputs={'goal': question})
    
    # Extract the contexts (the raw search results)
    contexts = []
    for tool_call in task_research.output.tool_calls:
        # The tool output is a JSON string, so we parse it
        tool_output = json.loads(tool_call.output)
        # We need to extract the 'content' from each search result
        for doc in tool_output:
            contexts.append(doc['content'])
            
    # Store the question, answer (report), and contexts
    results.append({
        "question": question,
        "answer": report,
        "contexts": contexts
    })

# 3. PREPARE THE DATASET FOR RAGAS
dataset = Dataset.from_list(results)

# 4. RUN THE EVALUATION
print("\n--- 📊 Running RAGAs Evaluation ---")
score = evaluate(
    dataset,
    metrics=[
        faithfulness,     # How factual is the report? (Does it stick to the contexts?)
        answer_relevancy  # How relevant is the report to the original question?
    ],
)

# 5. PRINT THE FINAL SCORE
print(json.dumps(score, indent=4))