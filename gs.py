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


def tagdoc(row, label_col='label', text_col='text'):
    """
    row: Series, a row off a data frame
    label_col: string, the name of the label column
    text_col: string, the name of the text column
    return: TaggedDocument
    """
    label = str(row[label_col])
    text = str(row[text_col])
    words = tokenize(text)
    return TaggedDocument(words, [label])


def tagdocs(df, label_col='label', text_col='text'):
    """
 
    Converts the data frame into a list of tagged lines. For each row in
    the data frame, the value of the text column is the document and
    the value of the label column is the label.

    df: DataFrame
    label_col: string, the name of the label column
    text_col: string, the name of the text column
    return: list of gensim TaggedDocument

    """
    return [tagdoc(row, label_col, text_col)
            for idx, row in df.iterrows()]


def listparams_sampling(dms=[0], sizes=[50], windows=[5], mincounts=[2],
                     samples=[1e-5], negatives=[5], hses=[0], workers=4, epochs=100):
    """
    return: a list of parameter combinations
    """
    return [{
        'dm': dm,
        'size': size,
        'window': window,
        'min_count': mincount,
        'sample': sample,
        'negative': neg,
        'hs': hs,
        'workers': workers,
        'iter': epochs
    } for neg in negatives for sample in samples for hs in hses
        for mincount in mincounts for window in windows
        for size in sizes for dm in dms]


def listparams(dms=[0], sizes=[50], windows=[5], mincounts=[2],
                     workers=4, epochs=100):
    """
    return: a list of parameter combinations
    """
    return [{
        'dm': dm,
        'size': size,
        'window': window,
        'min_count': mincount,
        'workers': workers,
        'iter': epochs
    } for mincount in mincounts for window in windows
        for size in sizes for dm in dms]


def train(taggeddocs, models):
    """
    taggeddocs: list of TaggedDocument
    models: list of models
    return: void
    """
    for model in models:
        model.build_vocab(taggeddocs)
        model.train(taggeddocs)


def topdocs(text, model, topn=10):
    """

    Finds the most similar docs (their labels) given the text and the model.

    text: string
    model: gensim model
    topn: integer
    return: list of tuples of (doc label, cosine similarity)

    """
    words = tokenize(text)
    vector = model.infer_vector(words)
    return model.docvecs.most_similar([vector], topn=topn)


def evaluate(df, model, label_col='label', text_col='text', topn=2):
    """

    Tests the model using the data frame.

    df: DataFrame
    label_col: string, the name of the label column
    text_col: string, the name of the text column
    model: gensim model to test
    topn: integer
    return: the list of labels, the list of booleans indicating hit

    """
    rows = [row for idx, row in df.iterrows()]
    labels = [str(row[label_col]) for row in rows]
    texts = [str(row[text_col]) for row in rows]
    toplists = [topdocs(text, model, topn) for text in texts]
    toplists = [set([label for label, score in toplist]) for toplist in toplists]
    hits = [label in toplist for label, toplist in zip(labels, toplists)]
    return labels, hits
