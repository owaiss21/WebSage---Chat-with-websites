# WebSage - Chat with Websites

This system enables users to chat with website content by leveraging Retrieval Augmented Generation (RAG). OpenAI embeddings convert website text into vectors, stored in ChromaDB, for efficient semantic retrieval. The retrieved information is then used to generate relevant responses.

**Technical Details:**

* **Embedding Generation:**
    * Utilizes OpenAI's model to transform website text into high-dimensional vector representations. These embeddings capture the semantic meaning of the text, allowing for similarity searches.
* **Vector Database:**
    * Employs ChromaDB as a vector store for efficient storage and retrieval of generated embeddings. ChromaDB's indexing capabilities enable rapid k-nearest neighbor (k-NN) searches.
* **Retrieval Augmented Generation (RAG):**
    * When a user submits a query, the system first generates an embedding for the query using OpenAI's embedding model.
    * This query embedding is then used to perform a similarity search within ChromaDB, retrieving the most relevant website text chunks.
    * The retrieved text chunks, along with the user's query, are passed to a large language model (LLM), such as GPT-3.5 or GPT-4, to generate a coherent and contextually relevant response.
* **Text Chunking:**
    * Website content is divided into smaller, manageable text chunks to improve retrieval accuracy and reduce LLM processing time.
    * Chunking strategies may vary depending on the website's structure and content.
* **API Integration:**
    * Integrates with OpenAI's API for embedding generation and language model interactions.

**User Interaction Details:**

* **Chat Interface:**
    * Provides a user-friendly chat interface where users can input their questions and receive answers.
* **Website Input:**
    * Users can provide the URL of the website they wish to interact with.
    * The system then automatically scrapes the website's text content.
* **Query Handling:**
    * The system processes user queries in real-time, retrieving relevant information and generating responses.
* **Contextual Awareness:**
    * Attempts to maintain conversational context, allowing users to ask follow-up questions.
* **Response Display:**
    * Displays generated responses in a clear and concise manner within the chat interface.
* **Error Handling:**
    * Provides informative error messages for invalid website URLs or other issues.
* **Feedback Mechanism(optional):**
    * Allows users to provide feedback on the accuracy and relevance of the generated responses.
