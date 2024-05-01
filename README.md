# Super-Accurate-RAG-based-Chatbot-for-Personal-Devices

## Inspiration
In a world where documents can be overwhelmingly long, the dream is to have your computer sift through the content for you. That's the vision behind the Super-Accurate-RAG-based-Chatbot-for-Personal-Devices - a tool designed to make life easier by digesting extensive information on your behalf.

## Unique Advantages
**Why choose this over others?**
- **Minimal Hardware**: Operates smoothly on just 8GB of RAM - no hefty hardware needed.
- **Contextual Clarity**: Unlike other chatbots that may struggle with context and produce irrelevant responses, this chatbot is engineered to provide precise and relevant answers.

## How It Works
1. **Topic Detection**: Identifies key topics within documents (currently in development).
2. **Intelligent Decision Making**: Determines whether to answer queries based on the document content or a web search.
3. **Data Retrieval**: Searches for relevance information within the document or on the web.
4. **Relevance Check**: Evaluates the relevance of the information retrieved.
5. **Answer Generation**: Crafts responses while self-checking for hallucinations and relevance.
6. **Answer Validation**: Delivers the answer if relevant; otherwise, it reinitiates the search from step 3.

## Frameworks Utilized
- Langchain's Langraph
- llama.cpp served using Ollama
- ChromaDB
- tavily (for web search)
- GPT4all (for document embeddings)

## Model
This chatbot is powered by Meta's LLama 3, 4-bit quantized to GGUF.

## See It in Action
Images showcasing the chatbot's capabilities can be found in the repository.

## TODO
- [ ] Add Topic Detection Capabilities
- [ ] Implement Graph RAG
- [ ] Implement MultiModal Capabilities
- [ ] Implement Conversation History
- [ ] Complete this README.md 😁
