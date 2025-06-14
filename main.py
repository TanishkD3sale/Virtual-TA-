# main.py
import os
import json
import sqlite3
import numpy as np
import re
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Tuple
import aiohttp
import asyncio
import logging
import base64
from fastapi.responses import JSONResponse
import uvicorn
import traceback
from dotenv import load_dotenv
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

# Configuration constants
class Config:
    DATABASE_FILE = "knowledge_base.db"
    RELEVANCE_THRESHOLD = 0.50
    MAX_SEARCH_RESULTS = 10
    ADJACENT_CHUNKS_COUNT = 4
    EMBEDDING_MODEL = "text-embedding-3-small"
    CHAT_MODEL = "gpt-4o-mini"
    API_ENDPOINT = "https://aipipe.org/openai/v1"
    
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("API_KEY")

config = Config()

# Data models and schemas
class SearchQuery(BaseModel):
    question: str
    image: Optional[str] = None

class SourceLink(BaseModel):
    url: str
    text: str

class SearchResult(BaseModel):
    answer: str
    links: List[SourceLink]

@dataclass
class ContentChunk:
    id: int
    content: str
    url: str
    similarity_score: float
    metadata: Dict[str, Any]

@dataclass
class DatabaseRow:
    id: int
    content: str
    url: str
    embedding_data: str
    additional_fields: Dict[str, Any]

# Abstract base classes for better architecture
class DatabaseManager(ABC):
    @abstractmethod
    async def initialize_connection(self):
        pass
    
    @abstractmethod
    async def fetch_content_chunks(self, table_name: str) -> List[DatabaseRow]:
        pass

class EmbeddingService(ABC):
    @abstractmethod
    async def create_embedding(self, text: str) -> List[float]:
        pass

class ResponseGenerator(ABC):
    @abstractmethod
    async def create_response(self, query: str, context_chunks: List[ContentChunk]) -> str:
        pass

# Concrete implementations
class SQLiteManager(DatabaseManager):
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_database_exists()
    
    def _ensure_database_exists(self):
        if not os.path.exists(self.db_path):
            self._create_initial_schema()
    
    def _create_initial_schema(self):
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.cursor()
            
            # Discourse content table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS discourse_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id INTEGER,
                topic_id INTEGER,
                topic_title TEXT,
                post_number INTEGER,
                author TEXT,
                created_at TEXT,
                likes INTEGER,
                chunk_index INTEGER,
                content TEXT,
                url TEXT,
                embedding BLOB
            )
            ''')
            
            # Documentation content table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS markdown_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_title TEXT,
                original_url TEXT,
                downloaded_at TEXT,
                chunk_index INTEGER,
                content TEXT,
                embedding BLOB
            )
            ''')
            
            connection.commit()
    
    async def initialize_connection(self):
        try:
            connection = sqlite3.connect(self.db_path)
            connection.row_factory = sqlite3.Row
            return connection
        except sqlite3.Error as error:
            log.error(f"Database connection failed: {error}")
            raise HTTPException(status_code=500, detail=f"Database error: {error}")
    
    async def fetch_content_chunks(self, table_name: str) -> List[DatabaseRow]:
        connection = await self.initialize_connection()
        try:
            cursor = connection.cursor()
            
            if table_name == "discourse_chunks":
                query = """
                SELECT id, post_id, topic_id, topic_title, post_number, 
                       author, created_at, likes, chunk_index, content, url, embedding 
                FROM discourse_chunks WHERE embedding IS NOT NULL
                """
            else:  # markdown_chunks
                query = """
                SELECT id, doc_title, original_url, downloaded_at, 
                       chunk_index, content, embedding 
                FROM markdown_chunks WHERE embedding IS NOT NULL
                """
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            database_rows = []
            for row in rows:
                if table_name == "discourse_chunks":
                    additional_fields = {
                        'post_id': row['post_id'],
                        'topic_id': row['topic_id'],
                        'topic_title': row['topic_title'],
                        'author': row['author'],
                        'chunk_index': row['chunk_index'],
                        'source_type': 'discourse'
                    }
                    url = self._normalize_discourse_url(row['url'])
                else:
                    additional_fields = {
                        'doc_title': row['doc_title'],
                        'chunk_index': row['chunk_index'],
                        'source_type': 'markdown'
                    }
                    url = self._normalize_markdown_url(row['original_url'], row['doc_title'])
                
                database_rows.append(DatabaseRow(
                    id=row['id'],
                    content=row['content'],
                    url=url,
                    embedding_data=row['embedding'],
                    additional_fields=additional_fields
                ))
            
            return database_rows
        finally:
            connection.close()
    
    def _normalize_discourse_url(self, url: str) -> str:
        if url and not url.startswith("http"):
            return f"https://discourse.onlinedegree.iitm.ac.in/t/{url}"
        return url or ""
    
    def _normalize_markdown_url(self, original_url: str, doc_title: str) -> str:
        if original_url and original_url.startswith("http"):
            return original_url
        return f"https://docs.onlinedegree.iitm.ac.in/{doc_title}"

class OpenAIEmbeddingService(EmbeddingService):
    def __init__(self, api_key: str, model: str = Config.EMBEDDING_MODEL):
        self.api_key = api_key
        self.model = model
        self.endpoint = f"{Config.API_ENDPOINT}/embeddings"
    
    async def create_embedding(self, text: str, retry_count: int = 3) -> List[float]:
        if not self.api_key:
            raise HTTPException(status_code=500, detail="API key not configured")
        
        for attempt in range(retry_count):
            try:
                log.info(f"Creating embedding (attempt {attempt + 1}) for text length: {len(text)}")
                
                headers = {
                    "Authorization": self.api_key,
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": self.model,
                    "input": text
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(self.endpoint, headers=headers, json=payload) as response:
                        if response.status == 200:
                            result = await response.json()
                            log.info("Embedding created successfully")
                            return result["data"][0]["embedding"]
                        elif response.status == 429:
                            wait_time = 5 * (attempt + 1)
                            log.warning(f"Rate limit hit, waiting {wait_time}s before retry")
                            await asyncio.sleep(wait_time)
                        else:
                            error_text = await response.text()
                            log.error(f"Embedding API error {response.status}: {error_text}")
                            if attempt == retry_count - 1:
                                raise HTTPException(status_code=response.status, detail=error_text)
                            
            except Exception as e:
                if attempt == retry_count - 1:
                    log.error(f"Final embedding attempt failed: {e}")
                    raise HTTPException(status_code=500, detail=str(e))
                await asyncio.sleep(3 * (attempt + 1))

class OpenAIResponseGenerator(ResponseGenerator):
    def __init__(self, api_key: str, model: str = Config.CHAT_MODEL):
        self.api_key = api_key
        self.model = model
        self.endpoint = f"{Config.API_ENDPOINT}/chat/completions"
    
    async def create_response(self, query: str, context_chunks: List[ContentChunk], retry_count: int = 2) -> str:
        if not self.api_key:
            raise HTTPException(status_code=500, detail="API key not configured")
        
        for attempt in range(retry_count):
            try:
                log.info(f"Generating response (attempt {attempt + 1}) for query")
                
                context_text = self._build_context_from_chunks(context_chunks)
                system_prompt = self._create_system_prompt()
                user_prompt = self._create_user_prompt(query, context_text)
                
                headers = {
                    "Authorization": self.api_key,
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.3
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(self.endpoint, headers=headers, json=payload) as response:
                        if response.status == 200:
                            result = await response.json()
                            log.info("Response generated successfully")
                            return result["choices"][0]["message"]["content"]
                        elif response.status == 429:
                            wait_time = 3 * (attempt + 1)
                            log.warning(f"Rate limit hit, waiting {wait_time}s")
                            await asyncio.sleep(wait_time)
                        else:
                            error_text = await response.text()
                            log.error(f"Response API error {response.status}: {error_text}")
                            if attempt == retry_count - 1:
                                raise HTTPException(status_code=response.status, detail=error_text)
                            
            except Exception as e:
                if attempt == retry_count - 1:
                    log.error(f"Final response generation attempt failed: {e}")
                    raise HTTPException(status_code=500, detail=str(e))
                await asyncio.sleep(2)
    
    def _build_context_from_chunks(self, chunks: List[ContentChunk]) -> str:
        context_parts = []
        for chunk in chunks:
            source_label = "Forum Discussion" if chunk.metadata.get('source_type') == 'discourse' else "Documentation"
            content_preview = chunk.content[:1500] if len(chunk.content) > 1500 else chunk.content
            context_parts.append(f"\n\n{source_label} (URL: {chunk.url}):\n{content_preview}")
        return "".join(context_parts)
    
    def _create_system_prompt(self) -> str:
        return ("You are an AI assistant that provides accurate answers based solely on provided context. "
                "If the context doesn't contain sufficient information, clearly state this limitation. "
                "Always include source references with exact URLs in your response.")
    
    def _create_user_prompt(self, query: str, context: str) -> str:
        return f"""Using only the provided context, answer this question: {query}

Context:
{context}

Format your response as:
1. A clear, comprehensive answer
2. Sources section listing URLs and relevant excerpts

Sources format:
Sources:
1. URL: [exact_url], Text: [brief relevant quote]
2. URL: [exact_url], Text: [brief relevant quote]

Ensure URLs are copied exactly as provided in the context."""

# Similarity calculation utilities
class SimilarityCalculator:
    @staticmethod
    def compute_cosine_similarity(vector_a: List[float], vector_b: List[float]) -> float:
        try:
            arr_a = np.array(vector_a)
            arr_b = np.array(vector_b)
            
            if np.all(arr_a == 0) or np.all(arr_b == 0):
                return 0.0
            
            dot_product = np.dot(arr_a, arr_b)
            magnitude_a = np.linalg.norm(arr_a)
            magnitude_b = np.linalg.norm(arr_b)
            
            if magnitude_a == 0 or magnitude_b == 0:
                return 0.0
            
            return float(dot_product / (magnitude_a * magnitude_b))
        except Exception as e:
            log.error(f"Similarity calculation error: {e}")
            return 0.0

# Content processing and search logic
class ContentProcessor:
    def __init__(self, db_manager: DatabaseManager, embedding_service: EmbeddingService):
        self.db_manager = db_manager
        self.embedding_service = embedding_service
        self.similarity_calc = SimilarityCalculator()
    
    async def process_multimodal_input(self, question: str, image_base64: Optional[str] = None) -> List[float]:
        if not image_base64:
            log.info("Processing text-only query")
            return await self.embedding_service.create_embedding(question)
        
        try:
            log.info("Processing multimodal query with image")
            image_description = await self._analyze_image_content(image_base64, question)
            enhanced_query = f"{question}\nImage context: {image_description}"
            return await self.embedding_service.create_embedding(enhanced_query)
        except Exception as e:
            log.error(f"Multimodal processing failed: {e}, falling back to text-only")
            return await self.embedding_service.create_embedding(question)
    
    async def _analyze_image_content(self, image_base64: str, question: str) -> str:
        headers = {"Authorization": config.api_key, "Content-Type": "application/json"}
        image_url = f"data:image/jpeg;base64,{image_base64}"
        
        payload = {
            "model": Config.CHAT_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Analyze this image in context of: {question}"},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{Config.API_ENDPOINT}/chat/completions", 
                                   headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    log.warning(f"Image analysis failed with status {response.status}")
                    return "Image analysis unavailable"
    
    async def find_relevant_content(self, query_embedding: List[float]) -> List[ContentChunk]:
        log.info("Searching for relevant content across data sources")
        
        all_chunks = []
        
        # Process discourse content
        discourse_rows = await self.db_manager.fetch_content_chunks("discourse_chunks")
        discourse_chunks = await self._convert_rows_to_chunks(discourse_rows, query_embedding)
        all_chunks.extend(discourse_chunks)
        
        # Process markdown content
        markdown_rows = await self.db_manager.fetch_content_chunks("markdown_chunks")
        markdown_chunks = await self._convert_rows_to_chunks(markdown_rows, query_embedding)
        all_chunks.extend(markdown_chunks)
        
        # Filter by relevance threshold
        relevant_chunks = [chunk for chunk in all_chunks 
                          if chunk.similarity_score >= Config.RELEVANCE_THRESHOLD]
        
        # Group and limit results
        grouped_chunks = self._group_chunks_by_source(relevant_chunks)
        final_chunks = self._select_top_chunks_per_source(grouped_chunks)
        
        log.info(f"Found {len(final_chunks)} relevant content chunks")
        return final_chunks[:Config.MAX_SEARCH_RESULTS]
    
    async def _convert_rows_to_chunks(self, rows: List[DatabaseRow], 
                                    query_embedding: List[float]) -> List[ContentChunk]:
        chunks = []
        for i, row in enumerate(rows):
            try:
                stored_embedding = json.loads(row.embedding_data)
                similarity = self.similarity_calc.compute_cosine_similarity(
                    query_embedding, stored_embedding
                )
                
                chunks.append(ContentChunk(
                    id=row.id,
                    content=row.content,
                    url=row.url,
                    similarity_score=similarity,
                    metadata=row.additional_fields
                ))
                
                if (i + 1) % 1000 == 0:
                    log.info(f"Processed {i + 1} chunks")
                    
            except Exception as e:
                log.error(f"Error processing row {row.id}: {e}")
        
        return chunks
    
    def _group_chunks_by_source(self, chunks: List[ContentChunk]) -> Dict[str, List[ContentChunk]]:
        groups = {}
        for chunk in chunks:
            if chunk.metadata.get('source_type') == 'discourse':
                key = f"discourse_{chunk.metadata.get('post_id')}"
            else:
                key = f"markdown_{chunk.metadata.get('doc_title')}"
            
            if key not in groups:
                groups[key] = []
            groups[key].append(chunk)
        
        return groups
    
    def _select_top_chunks_per_source(self, grouped_chunks: Dict[str, List[ContentChunk]]) -> List[ContentChunk]:
        selected_chunks = []
        for source_key, chunk_list in grouped_chunks.items():
            sorted_chunks = sorted(chunk_list, key=lambda x: x.similarity_score, reverse=True)
            selected_chunks.extend(sorted_chunks[:Config.ADJACENT_CHUNKS_COUNT])
        
        return sorted(selected_chunks, key=lambda x: x.similarity_score, reverse=True)
    
    async def enhance_chunks_with_context(self, chunks: List[ContentChunk]) -> List[ContentChunk]:
        log.info(f"Enhancing {len(chunks)} chunks with contextual information")
        
        connection = await self.db_manager.initialize_connection()
        try:
            enhanced_chunks = []
            cursor = connection.cursor()
            
            for chunk in chunks:
                enhanced_content = await self._get_adjacent_content(cursor, chunk)
                enhanced_chunk = ContentChunk(
                    id=chunk.id,
                    content=enhanced_content,
                    url=chunk.url,
                    similarity_score=chunk.similarity_score,
                    metadata=chunk.metadata
                )
                enhanced_chunks.append(enhanced_chunk)
            
            return enhanced_chunks
        finally:
            connection.close()
    
    async def _get_adjacent_content(self, cursor, chunk: ContentChunk) -> str:
        base_content = chunk.content
        adjacent_content = ""
        
        chunk_index = chunk.metadata.get('chunk_index', 0)
        
        if chunk.metadata.get('source_type') == 'discourse':
            post_id = chunk.metadata.get('post_id')
            
            # Get previous chunk
            if chunk_index > 0:
                cursor.execute(
                    "SELECT content FROM discourse_chunks WHERE post_id = ? AND chunk_index = ?",
                    (post_id, chunk_index - 1)
                )
                prev_result = cursor.fetchone()
                if prev_result:
                    adjacent_content = prev_result["content"] + " "
            
            # Get next chunk
            cursor.execute(
                "SELECT content FROM discourse_chunks WHERE post_id = ? AND chunk_index = ?",
                (post_id, chunk_index + 1)
            )
            next_result = cursor.fetchone()
            if next_result:
                adjacent_content += " " + next_result["content"]
        
        elif chunk.metadata.get('source_type') == 'markdown':
            doc_title = chunk.metadata.get('doc_title')
            
            # Get previous chunk
            if chunk_index > 0:
                cursor.execute(
                    "SELECT content FROM markdown_chunks WHERE doc_title = ? AND chunk_index = ?",
                    (doc_title, chunk_index - 1)
                )
                prev_result = cursor.fetchone()
                if prev_result:
                    adjacent_content = prev_result["content"] + " "
            
            # Get next chunk
            cursor.execute(
                "SELECT content FROM markdown_chunks WHERE doc_title = ? AND chunk_index = ?",
                (doc_title, chunk_index + 1)
            )
            next_result = cursor.fetchone()
            if next_result:
                adjacent_content += " " + next_result["content"]
        
        return f"{base_content} {adjacent_content}".strip() if adjacent_content else base_content

# Response parsing and formatting
class ResponseParser:
    @staticmethod
    def extract_answer_and_sources(llm_response: str) -> Dict[str, Any]:
        try:
            log.info("Parsing LLM response for answer and sources")
            
            # Try different source section headers
            source_headers = ["Sources:", "Source:", "References:", "Reference:"]
            answer_text = llm_response
            sources_text = ""
            
            for header in source_headers:
                if header in llm_response:
                    parts = llm_response.split(header, 1)
                    answer_text = parts[0].strip()
                    sources_text = parts[1].strip() if len(parts) > 1 else ""
                    break
            
            extracted_links = ResponseParser._parse_source_links(sources_text)
            
            log.info(f"Extracted answer ({len(answer_text)} chars) and {len(extracted_links)} source links")
            return {"answer": answer_text, "links": extracted_links}
            
        except Exception as e:
            log.error(f"Response parsing error: {e}")
            return {
                "answer": "Unable to parse the generated response properly.",
                "links": []
            }
    
    @staticmethod
    def _parse_source_links(sources_text: str) -> List[Dict[str, str]]:
        if not sources_text:
            return []
        
        links = []
        source_lines = [line.strip() for line in sources_text.split("\n") if line.strip()]
        
        for line in source_lines:
            # Remove list markers
            cleaned_line = re.sub(r'^\d+\.\s*|^-\s*|^\*\s*', '', line)
            
            # Extract URLs with flexible patterns
            url_patterns = [
                r'URL:\s*\[(.*?)\]',
                r'url:\s*\[(.*?)\]',
                r'\[(http[^\]]+)\]',
                r'URL:\s*(http\S+)',
                r'url:\s*(http\S+)',
                r'(http\S+)'
            ]
            
            url_found = None
            for pattern in url_patterns:
                match = re.search(pattern, cleaned_line, re.IGNORECASE)
                if match:
                    url_found = match.group(1).strip()
                    break
            
            # Extract descriptive text
            text_patterns = [
                r'Text:\s*\[(.*?)\]',
                r'text:\s*\[(.*?)\]',
                r'[""](.*?)[""]',
                r'Text:\s*"(.*?)"',
                r'text:\s*"(.*?)"'
            ]
            
            text_found = "Source reference"
            for pattern in text_patterns:
                match = re.search(pattern, cleaned_line, re.IGNORECASE)
                if match:
                    text_found = match.group(1).strip()
                    break
            
            if url_found and url_found.startswith("http"):
                links.append({"url": url_found, "text": text_found})
        
        return links

# Main application setup
class RAGApplication:
    def __init__(self):
        if not config.api_key:
            log.error("API_KEY environment variable not configured")
        
        self.db_manager = SQLiteManager(Config.DATABASE_FILE)
        self.embedding_service = OpenAIEmbeddingService(config.api_key)
        self.response_generator = OpenAIResponseGenerator(config.api_key)
        self.content_processor = ContentProcessor(self.db_manager, self.embedding_service)
        self.response_parser = ResponseParser()
    
    async def process_search_request(self, request: SearchQuery) -> Dict[str, Any]:
        try:
            log.info(f"Processing search request: '{request.question[:50]}...', has_image={request.image is not None}")
            
            if not config.api_key:
                raise HTTPException(status_code=500, detail="API key not configured")
            
            # Generate query embedding
            query_embedding = await self.content_processor.process_multimodal_input(
                request.question, request.image
            )
            
            # Find relevant content
            relevant_chunks = await self.content_processor.find_relevant_content(query_embedding)
            
            if not relevant_chunks:
                log.info("No relevant content found")
                return {"answer": "No relevant information found in the knowledge base.", "links": []}
            
            # Enhance with contextual information
            enhanced_chunks = await self.content_processor.enhance_chunks_with_context(relevant_chunks)
            
            # Generate response
            llm_response = await self.response_generator.create_response(request.question, enhanced_chunks)
            
            # Parse and format response
            parsed_result = self.response_parser.extract_answer_and_sources(llm_response)
            
            # Fallback: create links from chunks if parsing failed
            if not parsed_result["links"]:
                parsed_result["links"] = self._create_fallback_links(relevant_chunks[:5])
            
            log.info(f"Search completed: answer_length={len(parsed_result['answer'])}, links_count={len(parsed_result['links'])}")
            return parsed_result
            
        except Exception as e:
            log.error(f"Search processing error: {e}")
            log.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))
    
    def _create_fallback_links(self, chunks: List[ContentChunk]) -> List[Dict[str, str]]:
        links = []
        seen_urls = set()
        
        for chunk in chunks:
            if chunk.url not in seen_urls:
                seen_urls.add(chunk.url)
                preview = chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content
                links.append({"url": chunk.url, "text": preview})
        
        return links
    
    async def get_system_health(self) -> Dict[str, Any]:
        try:
            connection = await self.db_manager.initialize_connection()
            cursor = connection.cursor()
            
            # Count records in each table
            cursor.execute("SELECT COUNT(*) FROM discourse_chunks")
            discourse_total = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM markdown_chunks")
            markdown_total = cursor.fetchone()[0]
            
            # Count records with embeddings
            cursor.execute("SELECT COUNT(*) FROM discourse_chunks WHERE embedding IS NOT NULL")
            discourse_with_embeddings = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM markdown_chunks WHERE embedding IS NOT NULL")
            markdown_with_embeddings = cursor.fetchone()[0]
            
            connection.close()
            
            return {
                "status": "operational",
                "database_connection": "active",
                "api_key_configured": bool(config.api_key),
                "data_summary": {
                    "discourse_chunks": discourse_total,
                    "markdown_chunks": markdown_total,
                    "discourse_embeddings": discourse_with_embeddings,
                    "markdown_embeddings": markdown_with_embeddings
                }
            }
        except Exception as e:
            log.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "error_message": str(e),
                "api_key_configured": bool(config.api_key)
            }

# FastAPI application instance
app = FastAPI(
    title="Knowledge Base Search API",
    description="Advanced RAG-based search API for querying knowledge base content"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the RAG application
rag_app = RAGApplication()

# API endpoints
@app.post("/api", response_model=SearchResult)
async def search_knowledge_base(request: SearchQuery):
    result = await rag_app.process_search_request(request)
    return SearchResult(**result)

@app.get("/health")
async def health_check():
    health_data = await rag_app.get_system_health()
    if health_data["status"] == "error":
        return JSONResponse(status_code=500, content=health_data)
    return health_data

# Application entry point
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",  # Changed from 127.0.0.1
        port=port,       # Use Render's PORT environment variable
        reload=False,    # Disable reload in production
        log_level="info"
    )
