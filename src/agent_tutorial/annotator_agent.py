from pydantic import BaseModel
from pydantic_ai import Agent
from typing import List, Optional
import click
import sys
from oaklib import get_adapter
from agent_tutorial.oak_agent import search_go

class TextAnnotation(BaseModel):
    """
    A text annotation is a span of text and the GO ID and label for the biological processes or cellular component it mentions.
    Use `text` for the source text, and `go_id` and `go_label` for the GO ID and label of the anatomical structure in the ontology.
    """
    text: str
    go_id: Optional[str] = None
    go_label: Optional[str] = None

class TextAnnotationResult(BaseModel):
    annotations: List[TextAnnotation]

annotator_agent = Agent(  
    #'claude-3-7-sonnet-latest',
    'openai:gpt-4o',
    system_prompt="""
    Extract all go terms from the text. Return the as a list of annotations.
    Be sure to include all spans mentioning biological processes or cellular components; if you cannot
    find a GO ID, then you should still return a TextAnnotation, just leave
    the go_id field empty.

    However, before giving up you should be sure to try different combinations of
    synonyms with the `search_go` tool.
    When searching for synonyms, try substituting individual words or phrases in the span
    for synonymous words or phrases, e.g. you might substitute regulation for control.
    """,
    tools=[search_go],
    result_type=TextAnnotationResult,  
)
DEFAULT_TEXT = ''

@click.command()
@click.argument('text', required=False)
def main(text: str = None):
    """Run the annotator agent on the given text or from STDIN."""
    if text is None:
        text = sys.stdin.read().strip()
    
    if not text:
        click.echo("Error: No text provided via argument or STDIN", err=True)
        sys.exit(1)
    
    result = annotator_agent.run_sync(text)
    print("## Result:")
    for a in result.output.annotations:
        print(f"  {a.text} ==> {a.go_id} {a.go_label}")

if __name__ == "__main__":
    main()
