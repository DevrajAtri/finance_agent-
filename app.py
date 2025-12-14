import streamlit as st
import os
import tempfile
from typing import List, Tuple

from state import AgentState
from graph import app as agent_app  # Importing the compiled graph
from ingest_user_file import ingest_user_file, IngestInput

st.set_page_config(page_title="Financial Agent", layout="wide")
st.title("🤖 AI Financial Analyst Agent")

# We need to store the chat history and the file info so they persist 
# when Streamlit re-runs the script.

if "messages" not in st.session_state:
    st.session_state.messages = []  # Store UI chat history

if "user_file_info" not in st.session_state:
    st.session_state.user_file_info = None # Store uploaded file metadata

with st.sidebar:
    st.header("📂 Data Source")
    st.write("Upload a PDF/CSV/TXT to analyze.")
    
    uploaded_file = st.file_uploader("Upload File", type=["pdf", "csv", "txt"])
    
    # Logic to handle the upload only once
    if uploaded_file and not st.session_state.user_file_info:
        with st.spinner("Ingesting file... (Vectors + Pinecone)"):
            
            # Save to a temporary file because your ingest function expects a path
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            #  CALL YOUR BACKEND INGESTION 
            ingest_input = IngestInput(
                file_path=tmp_path,
                file_name=uploaded_file.name
            )
            result = ingest_user_file(ingest_input)
            
            # Store the result (Namespace ID) in session state
            if result.get("user_file_info"):
                st.session_state.user_file_info = result["user_file_info"]
                st.success(f"Loaded: {uploaded_file.name}")
            else:
                st.error("Ingestion failed.")
            
            # Cleanup temp file
            os.remove(tmp_path)

    # Show current status
    if st.session_state.user_file_info:
        st.info(f"Using File: {st.session_state.user_file_info.get('filename')}")
        if st.button("Clear File"):
            st.session_state.user_file_info = None
            st.rerun()



# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if prompt := st.chat_input("Ask about financial data..."):
    
    # 1. Add user message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Prepare the Agent State
    # We reconstruct the state with the current query and history
    initial_state = AgentState(
        query=prompt,
        chat_history=[(m["role"], m["content"]) for m in st.session_state.messages],
        user_file_info=st.session_state.user_file_info, # Pass the file ID!
        retrieved_chunks=[],
        tool_calls=[],
        tool_outputs=[],
        clarification_question=None,
        final_answer=None
    )

    # 3. Run the Agent (Stream the steps)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.status("Agent is working...", expanded=True) as status:
            
            # Stream the graph execution
            # recursion_limit protects against infinite loops
            for step in agent_app.stream(initial_state, {"recursion_limit": 20}):
                
                step_name = list(step.keys())[0]
                step_state = step[step_name]
                
                if step_name == "router":
                    status.write("🧠 Routing query...")
                elif step_name == "retrieve":
                    n_chunks = len(step_state.get('retrieved_chunks', []))
                    status.write(f"🔍 Retrieved {n_chunks} text chunks.")
                elif step_name == "tool_planner":
                    status.write("📋 Planning calculations...")
                elif step_name == "tool_executor":
                    status.write("⚙️ Executing tools...")
                elif step_name == "ask_user":
                    # If the agent asks a question, we treat it as the response
                    full_response = step_state.get("clarification_question")
                    status.update(label="❓ Need Clarification", state="complete")
                    break # Stop streaming
                elif step_name == "synthesizer":
                    # This is the final answer
                    full_response = step_state.get("final_answer")
                    status.update(label="✅ Finished", state="complete")
        
        # 4. Display Final Answer
        message_placeholder.markdown(full_response)
        
        # 5. Save assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": full_response})