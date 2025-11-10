# src/lazifetch/model/Result.py


class Result:
    def __init__(
        self, title="", abstract="", article=None, citations_count=0, year=None
    ) -> None:
        self.title = title
        self.abstract = abstract
        self.article = article
        self.citations_count = citations_count
        self.year = year
