# web_app.py
import streamlit as st
from main import crew  # Import your crew from main.py
from tasks import task_research # Import the research task

# --- Page Configuration ---
st.set_page_config(
    page_title="Autonomous Market Research Agent",
    page_icon="🤖",
    layout="wide",
)

# --- App Title ---
st.title("Autonomous Market Research Agent 🤖")

# --- Sidebar for Inputs ---
st.sidebar.header("Research Goal")
goal_input = st.sidebar.text_area(
    "Enter your high-level research goal:",
    height=200,
    placeholder="e.g., Give me a competitive analysis of Tesla vs. Rivian, focusing on 2025 market sentiment, battery technology, and production numbers."
)

# --- Main App Body ---
st.header("Your AI-Generated Report")

if st.sidebar.button("Start Research Crew"):
    if not goal_input:
        st.sidebar.error("Please enter a research goal to begin.")
    else:
        # Show a loading spinner while the crew is working
        with st.spinner("Your AI crew is on the job! This may take a few minutes..."):
            try:
                # 1. Create the inputs dictionary for the crew
                inputs = {'goal': goal_input}
                
                # 2. Kick off the crew's work
                report = crew.kickoff(inputs=inputs)
                
                # 3. Display the final report
                st.subheader("Final Report:")
                st.markdown(report)
                
                # --- THE FIX ---
                # We display the raw string output from the researcher,
                # not a non-existent '.tool_calls' attribute.
                with st.expander("Show Raw Research Data (from Researcher)"):
                    st.text(task_research.output.raw)
                # --- END FIX ---

            except Exception as e:
                st.error(f"An error occurred: {e}")
else:
    st.info("Enter your research goal in the sidebar and click 'Start Research Crew' to begin.")