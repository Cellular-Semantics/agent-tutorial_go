from typing import List, Tuple, Optional
import click
from oaklib import get_adapter
from pydantic import BaseModel
from pydantic_ai import Agent

oak_agent = Agent(  
    'openai:gpt-4o',
    system_prompt="""
    You are an expert ontology curator. Use the ontologies at your disposal to
    answer the users questions.
    """,  
)

@oak_agent.tool_plain
def search_go(term: str) -> List[Tuple[str, str]]:
    """
    Search the GO ontology for a term.

    Note that search should take into account synonyms, but synonyms may be incomplete,
    so if you cannot find a concept of interest, try searching using related or synonymous
    terms.

    If you are searching for a composite term, try searching on the sub-terms to get a sense
    of the terminology used in the ontology.

    Args:
        term: The term to search for.

    Returns:
        A list of tuples, each containing an GO ID and a label.
    """
    adapter = get_adapter("ols:go")
    results = adapter.basic_search(term)
    labels = list(adapter.labels(results))
    print(f"## Query: {term} -> {labels}")
    return labels


@click.command()
@click.argument('query', default='CTSH is involved in surfactant homeostasis')
def main(query: str):
    """Run the oak agent with the given query."""
    result = oak_agent.run_sync(query)
    print(result.data)


if __name__ == "__main__":
    main()

