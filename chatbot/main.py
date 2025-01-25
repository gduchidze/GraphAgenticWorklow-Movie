import json
import os
from typing import TypedDict, Annotated, Union, Dict, List
from langchain_core.agents import AgentAction
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
import operator
from langchain_openai import ChatOpenAI
import logging
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from duckduckgo_search import DDGS


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

class AgentState(TypedDict):
   input: str
   chat_history: list[BaseMessage]
   intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]


@tool("Search-Movies")
def search_movies(query: str):
   """
   Search movies from the IMDB database using vector similarity.
   """
   try:
      embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

      vector_store = PineconeVectorStore(
         embedding=embeddings,
         index_name="movies-database",
         pinecone_api_key=os.getenv("PINECONE_API_KEY"),
         text_key="text"
      )

      results = vector_store.similarity_search(query, k=3)

      processed = []
      for doc in results:
         content = f"Title: {doc.metadata['title']}\n"
         content += f"Year: {doc.metadata['year']}\n"
         content += f"Rating: {doc.metadata['rating']}\n"
         content += f"Genre: {doc.metadata['genre']}\n"
         content += f"Director: {doc.metadata['director']}\n"
         content += f"Stars: {doc.metadata['stars']}"

         processed.append({
            "content": content,
            "meta_score": doc.metadata["meta_score"],
            "gross": doc.metadata["gross"]
         })

      return processed if processed else "No results found"

   except Exception as e:
      logger.error(f"Search error: {str(e)}")
      return f"Search failed: {str(e)}"


@tool("Realtime Movie Critic Search")
def realtime_movie_critic_search(movie_title: str) -> Union[List[Dict], str]:
    """
    Perform real-time search for critics' opinions from reputable sources.

    Example Input: "The Dark Knight reviews"
    Example Output: List of review snippets with sources or error message
    """
    try:
        query = f"{movie_title} site:rottentomatoes.com OR site:metacritic.com OR site:imdb.com"
        results = DDGS().text(query, max_results=5)

        reviews = []
        for result in results:
            if any(keyword in result.get('title', '').lower()
                   for keyword in ['review', 'critic', 'analysis']):
                reviews.append({
                    "source": result.get('href', ''),
                    "summary": result.get('body', '')[:200] + "...",
                })

        return reviews[:2] if reviews else "No recent reviews found"

    except Exception as e:
        logger.error(f"Critic search error: {str(e)}", exc_info=True)
        return f"Review search failed: {str(e)}"


@tool("Plot Analysis")
def plot_analysis(movie_title: str) -> Union[Dict, str]:
    """
    Analyze movie plot using GPT-4 with structured output.

    Example Input: "Inception"
    Example Output: Structured analysis or error message
    """
    try:
        movie_data = search_movies(movie_title)
        if isinstance(movie_data, str) or not movie_data:
            return "Movie not found in database"

        plot = movie_data[0].get("overview", "")
        if not plot:
            return "No plot available for analysis"

        chat_openai = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

        analysis = chat_openai.predict(
            f"""Analyze this movie plot and provide structured insights:
            ---
            {plot}
            ---
            Provide analysis in this format:
            {{
                "themes": ["list", "of", "main", "themes"],
                "character_development": "brief analysis",
                "narrative_structure": "analysis of storytelling approach",
                "unique_elements": ["list", "of", "unique", "aspects"]
            }}"""
        )

        try:
            return json.loads(analysis.strip("```json\n").rstrip("\n```"))
        except json.JSONDecodeError:
            return analysis

    except Exception as e:
        logger.error(f"Plot analysis error: {str(e)}", exc_info=True)
        return f"Analysis failed: {str(e)}"