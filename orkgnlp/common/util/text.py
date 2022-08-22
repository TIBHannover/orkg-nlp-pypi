""" Utility file for text processing functionalities. """
import re
import string
import nltk

from typing import Union, List
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from orkgnlp.common.util.decorators import sanitize_text


@sanitize_text
def remove_punctuation(text: str) -> str:
    """
    Removes punctuations from the given ``text``.

    :param text:
    :return: text with punctuations removed.
    """
    return text.translate(str.maketrans('', '', string.punctuation))


@sanitize_text
def remove_stopwords(text: str, language: str = 'english') -> str:
    """
    Removes stopwords from the given ``text``.
    Uses the stopwords defined in ``NLTK``.

    :param text:
    :param language: defaults to ``english``.
    :return: text with stopwords removed.
    """
    nltk.download('stopwords', quiet=True)

    return ' '.join(
        [word for word in text.split() if word not in stopwords.words(language)]
    )


@sanitize_text
def lemmatize(text: str) -> str:
    """
    Lemmatizes the words in the given ``text``.
    Uses the WordNetLemmatizer defined in ``NLTK``.

    :param text:
    :return: text with its lemmatized words.
    """
    nltk.download('wordnet', quiet=True)

    stemmer = WordNetLemmatizer()
    return ' '.join(
        [stemmer.lemmatize(word) for word in text.split()]
    )


@sanitize_text
def replace(text: str, chars: Union[str, List[str]], replacement: str) -> str:
    """
    Replaces the occurrences of any item in ``chars`` in ``text`` with ``replacement``.

    :param text:
    :param chars: Regex or list of regexes that will be separated with | as OR operator,
        their occurrences to be replaced.
    :param replacement: The replacement string
    """
    chars = chars if isinstance(chars, list) else [chars]
    regex = '|'.join(chars)
    return re.sub(regex, replacement, text)


@sanitize_text
def trim(text: str) -> str:
    """
    Removes any whitespace character and returns the same input only with single spaces.

    :param text:
    """
    return ' '.join(text.split())
