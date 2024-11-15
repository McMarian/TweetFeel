# utils.py

import re

def remove_emojis(text):
    """
    Remove emojis and other non-text characters from text.

    Parameters:
    - text: The text from which to remove emojis.

    Returns:
    - Text without emojis.
    """
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # Emoticons
        u"\U0001F300-\U0001F5FF"  # Symbols & Pictographs
        u"\U0001F680-\U0001F6FF"  # Transport & Map Symbols
        u"\U0001F1E0-\U0001F1FF"  # Flags
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)
