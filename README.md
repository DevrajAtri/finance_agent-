Financial Analysis Agent (RAG + Agentic Workflow)
An advanced AI financial analyst capable of performing complex valuation, retrieving company-specific data, and executing multi-step financial reasoning.

Built with LangGraph and Google Gemini, this agent solves common reliability issues in LLM applications (like infinite loops and lazy tool calling) through a robust Router-Planner-Executor architecture.

**Key Features
Agentic Architecture: Uses a specialized Router to distinguish between general chat, document retrieval (RAG), and tool execution.

**Robust Tooling:

Valuation Engine: Calculates NPV, WACC, CAPM, and Terminal Value.

Ratio Calculator: Computes P/E, ROE, Current Ratio, and margins on the fly.

Live Data Fetching: Retrives real-time stock prices (Ticker-ready).

Production-Grade Reliability:

"Lazy LLM" Fix: Implements a JSON-string bypass pattern to ensure the Planner never returns empty arguments.

Anti-Loop Logic: The Router detects failed execution attempts and short-circuits to prevent infinite retry loops.

Defensive Execution: Tool failures are caught and logged without crashing the entire agent graph.

Automated Testing: Includes a comprehensive tester.py suite with "LLM-as-a-Judge" grading and rate-limit cooldowns.

**Tech Stack
Framework: LangChain & LangGraph

LLM: Google Gemini 2.5 Flash 

Interface: Streamlit

Vector Store: Pinecone

Language: Python 3.11

****
Start the Streamlit app to interact with the agent visually.

type in terminal:-

streamlit run app.py

****
few precautions:-
1. the modes used for embedding and searching are locally downloaded , so before running the code dowload them and set the path 
2. models are :- bge-m3 , bge-reranker-large , naver splade 
3. make sure you have api keys for gemini and pinecone 
