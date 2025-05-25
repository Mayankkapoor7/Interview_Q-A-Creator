# Interview_Q-A-Creator

This app allows you to upload a PDF (e.g., a resume, report, or text-heavy document) and automatically generates possible interview questions along with detailed answers extracted and refined from the content. The output is displayed interactively on the web interface, and you can download the Q&A as a formatted Word document.

Features
PDF Upload: Easy-to-use drag and drop or file select for PDFs up to 5 pages.

Question Generation: Extracts relevant interview questions using GPT-3.5 turbo and LangChain chains.

Answer Retrieval: Uses vector embeddings and a retrieval-based QA chain to answer each generated question.

Clean Q&A Display: Shows questions and answers one after the other for clarity.

Download Option: Export the generated Q&A as a nicely formatted Word (.docx) document.

Powered by LangChain: Modular, flexible NLP pipeline combining summarization, embedding, and retrieval.

Built with Streamlit: Simple, fast UI for easy interaction and deployment.

How It Works
Load PDF & Extract Text: Uses PyPDFLoader to read PDF pages.

Split Text for Question Generation: Splits large content into manageable chunks for GPT-3.5 summarization chains.

Generate Questions: Uses a LangChain summarization/refinement chain to generate interview questions.

Create Embeddings & Retriever: Builds a FAISS vector store from text chunks to retrieve context for each question.

Answer Questions: RetrievalQA chain fetches relevant info and generates answers.

Display & Export: Shows questions and answers on the UI and creates a downloadable Word document.
