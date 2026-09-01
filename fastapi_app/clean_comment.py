import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)



STOP_WORDS = set(stopwords.words("english")) - {"not", "but", "however", "no", "yet"}
LEMMATIZER = WordNetLemmatizer()

def preprocess_comment(comment: str) -> str:
    try:
        comment = comment.lower()
        comment = comment.strip()
        comment = re.sub(r"\n", " ", comment)
        comment = re.sub(r"[^A-Za-z0-9\s!?.,]","",comment)
        comment = " ".join(word for word in comment.split() if word not in STOP_WORDS )
        comment = " ".join(LEMMATIZER.lemmatize(word) for word in comment.split() )
        return comment
    except Exception as e:
        return comment