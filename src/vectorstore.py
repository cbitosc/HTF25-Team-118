import cohere
import fitz  # PyMuPDF
from pinecone import Pinecone, ServerlessSpec
import time

class VectorStore:
    def __init__(self, cohere_api_key: str, pinecone_api_key: str):
        """
        Initializes the clients for Cohere and Pinecone.
        """
        self.co = cohere.Client(cohere_api_key)
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index_name = 'rag-qa-bot'
        self.index = None
        self.retrieve_top_k = 10
        self.rerank_top_k = 3
        
        # Get the embedding dimension from the Cohere model
        try:
            response = self.co.embed(
                texts=["hello world"],
                model="embed-english-v3.0",
                input_type="search_document"
            )
            self.dimension = len(response.embeddings[0])
        except Exception as e:
            print(f"Error getting embedding dimension, using default 1024: {e}")
            self.dimension = 1024 # Fallback dimension for embed-english-v3.0

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extracts raw text from a PDF file.
        """
        text = ""
        with fitz.open(pdf_path) as pdf:
            for page_num in range(pdf.page_count):
                page = pdf.load_page(page_num)
                text += page.get_text("text")
        return text

    def _split_text(self, text: str, chunk_size=1000) -> list[str]:
        """
        Splits the text into chunks based on sentences and a max chunk size.
        """
        chunks = []
        sentences = text.split(". ")
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def _embed_chunks(self, chunks: list[str], batch_size=90) -> list[list[float]]:
        """
        Embeds a list of text chunks in batches.
        """
        embeddings = []
        total_chunks = len(chunks)
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:min(i + batch_size, total_chunks)]
            batch_embeddings = self.co.embed(
                texts=batch, input_type="search_document", model="embed-english-v3.0"
            ).embeddings
            embeddings.extend(batch_embeddings)
        return embeddings

    def _index_chunks(self, chunks: list[str], embeddings: list[list[float]]):
        """
        Initializes the Pinecone index (if not present) and upserts the vectors.
        """
        if self.index_name not in self.pc.list_indexes().names():
            print(f"Creating index '{self.index_name}'...")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            # Wait for index to be ready
            while not self.pc.describe_index(self.index_name).status['ready']:
                print("Waiting for index to be ready...")
                time.sleep(1)
        
        self.index = self.pc.Index(self.index_name)
        
        # Clear the index before upserting new data
        print("Clearing old data from index...")
        self.index.delete(delete_all=True)

        # Prepare vectors in the correct format for upsert
        vectors_to_upsert = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vectors_to_upsert.append({
                "id": str(i),
                "values": embedding,
                "metadata": {"text": chunk}
            })
        
        # Upsert in batches
        print(f"Upserting {len(vectors_to_upsert)} vectors...")
        batch_size = 100  # Pinecone recommends batches of 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            self.index.upsert(vectors=batch)
        
        print(f"Indexing complete. Index stats: {self.index.describe_index_stats()}")

    def process_document(self, pdf_path: str):
        """
        Public method to orchestrate the full ingestion pipeline.
        """
        print("Starting document processing...")
        # 1. Extract text
        pdf_text = self._extract_text_from_pdf(pdf_path)
        print(f"Extracted {len(pdf_text)} characters.")
        
        # 2. Split text
        chunks = self._split_text(pdf_text)
        print(f"Split text into {len(chunks)} chunks.")
        if not chunks:
            print("No chunks created. Check PDF content.")
            return

        # 3. Embed chunks
        embeddings = self._embed_chunks(chunks)
        print(f"Created {len(embeddings)} embeddings.")
        if not embeddings:
            print("No embeddings created. Check Cohere API key.")
            return

        # 4. Index chunks
        self._index_chunks(chunks, embeddings)
        print("Document processing finished.")

    def retrieve(self, query: str) -> list[dict]:
        """
        Retrieves and reranks relevant document chunks for a given query.
        Returns a list of metadata dictionaries (which co.chat expects).
        """
        if not self.index:
            print("Error: Index is not initialized. Please process a document first.")
            return []

        # 1. Embed the query
        query_emb_response = self.co.embed(
            texts=[query], model="embed-english-v3.0", input_type="search_query"
        )
        query_emb = query_emb_response.embeddings[0]
        
        # 2. Retrieve from Pinecone
        res = self.index.query(vector=query_emb, top_k=self.retrieve_top_k, include_metadata=True)
        
        docs_to_rerank = [match['metadata']['text'] for match in res['matches']]
        
        if not docs_to_rerank:
            return []

        # 3. Rerank with Cohere
        rerank_results = self.co.rerank(
            query=query,
            documents=docs_to_rerank,
            top_n=self.rerank_top_k,
            model="rerank-english-v3.0"  # <-- THIS IS THE FIX
        )
        
        # Return the reranked documents in the format co.chat expects
        reranked_docs_for_chat = []
        for result in rerank_results.results:
            # We need to find the original metadata
            original_metadata = res['matches'][result.index]['metadata']
            reranked_docs_for_chat.append(original_metadata)
            
        return reranked_docs_for_chat

