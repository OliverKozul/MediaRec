import pandas as pd
from typing import Union

def space_tokenizer(text: str) -> list:
    """Tokenizes text by splitting on spaces."""
    return text.split()

def identity_preprocessor(text: str) -> str:
    """Returns the text unchanged."""
    return text

def process_actors(actors: str, max_actors: int = 3) -> str:
    """Process actors string: lowercase, remove spaces, limit to first N actors, join with comma."""
    if not isinstance(actors, str):
        return ''
    return ', '.join([
        ''.join(name.strip().lower().split(' '))
        for name in actors.split(',')[:max_actors]
    ])

def process_director(director: str) -> str:
    """Process director string: lowercase, remove spaces."""
    if not isinstance(director, str):
        return ''
    return ''.join(director.strip().lower().split(' '))

def process_genres(genres: str, max_genres: int = 2) -> str:
    """Process genres string: limit to first N genres, join with pipe."""
    if not isinstance(genres, str):
        return ''
    return '|'.join(genres.split('|')[:max_genres])

def process_media_dataset(media: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
    """Preprocess media data for consistent formatting. Handles both DataFrame and Series (row)."""
    if isinstance(media, pd.DataFrame):
        media['actors'] = media['actors'].apply(process_actors)
        media['director'] = media['director'].apply(process_director)
        media['genres'] = media['genres'].apply(process_genres)
        return media
    elif isinstance(media, pd.Series):
        media['actors'] = process_actors(media['actors'])
        media['director'] = process_director(media['director'])
        media['genres'] = process_genres(media['genres'])
        return media
    else:
        return media