""" Utility file for text processing functionalities. """
import string

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


def remove_punctuation(text: str) -> str:
    """
    Removes punctuations from the given ``text``.

    :param text:
    :return: text with punctuations removed.
    """
    return text.translate(str.maketrans('', '', string.punctuation))


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
