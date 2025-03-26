# WebSage: Website Conversation Tool

WebSage allows users to interact with website content through a conversational interface. It uses Retrieval Augmented Generation (RAG) to provide answers derived directly from the website's information.

**How it Works:**

* **Semantic Indexing:**
    * Website text is converted into vector representations using OpenAI's embedding models. These vectors capture the meaning of the text, enabling efficient semantic search.
* **Vector Database:**
    * ChromaDB is used to store and manage these vector representations, allowing for fast retrieval of relevant information.
* **Response Generation:**
    * When a user asks a question, the system retrieves the most relevant text segments from the website using vector similarity search.
    * These segments, along with the user's question, are then used to generate a response through a large language model.

**Key Features:**

* **Direct Website Interaction:**
    * Users can ask questions about the content of any provided website URL.
* **Contextual Responses:**
    * The system aims to provide answers that are directly relevant to the website's content and the user's query.
* **Efficient Information Retrieval:**
    * Vector databases are used to quickly find the most relevant information within the website's text.
* **Clear Output:**
    * Responses are designed to be easy to read and understand.
* **Error Handling:**
    * Informative error messages are provided for invalid inputs.

**Technical Overview:**

The system leverages OpenAI's embedding API for vector generation and a large language model for response creation. ChromaDB is used as the vector storage and retrieval mechanism. Website text is chunked to improve retrieval accuracy.

**Contributions:**

Contributions to this project are welcome. Please submit pull requests with any improvements or bug fixes.
