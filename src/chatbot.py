import cohere
import uuid

class Chatbot:
    def __init__(self, vectorstore, cohere_api_key: str):
        """
        Initializes the chatbot with a vectorstore and Cohere client.
        """
        self.vectorstore = vectorstore
        self.co = cohere.Client(cohere_api_key)
        self.conversation_id = str(uuid.uuid4())

    def respond(self, user_message: str):
        """
        Generates a response to the user's message using the RAG pipeline.
        Returns a generator stream from co.chat_stream.
        """
        
        # 1. Generate search queries.
        # The 'search_queries_only' parameter has been deprecated by Cohere.
        # We will now use the user's message directly as the search query.
        search_queries = [user_message]
        print(f"Using search query: {search_queries}")
            
        # 2. Retrieve documents for the query
        retrieved_docs_metadata = []
        for query in search_queries:
            retrieved_docs_metadata.extend(self.vectorstore.retrieve(query))

        # 3. De-duplicate the retrieved documents
        # The 'documents' parameter for co.chat expects a list of dictionaries,
        # e.g., [{"text": "doc1"}, {"text": "doc2"}]
        unique_docs = {}
        for doc in retrieved_docs_metadata:
            # Use the 'text' content as the key for de-duplication
            if doc['text'] not in unique_docs:
                unique_docs[doc['text']] = doc
        
        final_documents = list(unique_docs.values())
        
        if final_documents:
            print(f"Retrieved {len(final_documents)} unique documents for context.")
        else:
            print("No documents retrieved from vector store.")

        # 4. Generate the final response using the retrieved documents
        # We return the stream generator directly
        response_stream = self.co.chat_stream(
            message=user_message,
            model="command-r-08-2024",  # <-- FIX 1: Updated model
            documents=final_documents,  
            conversation_id=self.conversation_id,
        )
        
        return response_stream

