from string import punctuation
from nltk import word_tokenize, download
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download the Punctuation
download('punkt')
download('stopwords')

# Instantiate the PorterStemmer
ps = PorterStemmer()

def transform_text(text: str) -> str:
    """
    Inputs:
        - text (string): The input text portaining to a SMS or email to tokenize and cleanup before getting a prediction.

    Output:
        - stemmed_text (string)

    """

    # Making the text lowercase
    text = text.lower()

    # Applying tokenization at the word level via nltk
    text = word_tokenize(text)


    y = []
    
    # Iterate through the characters in the text
    for i in text:
        # Check if it's alphanumeric
        ## Validating against non-UTF-8 characters
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # Filter out stopwords and punctuation characters
    for i in text:
        if i not in stopwords.words('english') and i not in punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # Applying PorterStemmer to the tokens
    for i in text:
        y.append(ps.stem(i))

    # Getting the cleaned up text as an output
    return " ".join(y)