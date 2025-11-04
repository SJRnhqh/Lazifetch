# src/Lazifetch/model/Result.py


class Result:
    """
    Result class for a paper
        Attributes:
            title: title of the paper
            abstract: abstract of the paper
            article: article of the paper
            citations_count: citations count of the paper
            year: year of the paper
    """

    def __init__(
        self, title="", abstract="", article=None, citations_count=0, year=None
    ) -> None:
        self.title = title
        self.abstract = abstract
        self.article = article
        self.citations_count = citations_count
        self.year = year
