from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess


def tokenize(text):
    """

    Breaks text into tokens. Currently uses gensim's simple_preprocess()
    which splits on whitespace, converts to lower case, ingores numbers.
    Note this tokenizer is less ideal as it is too agressive against
    numbers. We need a better tokenizer for medical codes/terms/concepts.

    text: string, the input text
    return: list of strings, the generated tokens

    """
    return simple_preprocess(text)


def tag_docs(df, label_col='label', text_col='text'):
    """
 
    Converts the data frame into a list of tagged lines. For each row in
    the data frame, the value of the text column is the document and
    the value of the label column is the label.

    df: DataFrame
    label_col: string, the name of the label column
    text_col: string, the name of the text column
    return: list of gensim TaggedDocument

    """
    tagged_lines = []
    for idx, row in df.iterrows():
        label = str(row[label_col])
        text = str(row[text_col])
        tagged_lines.append(
            TaggedDocument(simple_preprocess(text), [label]))
    return tagged_lines


def top_docs(text, model, topn=10):
    """

    Finds the most similar docs (their labels) given the text and the model.

    text: string
    model: gensim model
    topn: integer
    return: list of tuples of (doc label, cosine similarity)

    """
    words = tokenize(text)
    return model.docvecs.most_similar(
            [model.infer_vector(words)], topn=topn)


def test_model(df, model, label_col='label', text_col='text', topn=2):
    """

    Tests the model using the data frame.
    
    df: DataFrame
    label_col: string, the name of the label column
    text_col: string, the name of the text column
    model: gensim model to test
    topn: integer
    return: (total, hits)

    """
    total, hits = 0, 0
    for idx, row in df.iterrows():
        label = row[label_col]
        text = row[text_col]
        top_list = top_docs(text, model, topn)
        total = total + 1
        for item in top_list:
            if item[0] == label:
                hits = hits + 1
                break
    return total, hits
