# model/Result.py


class Result:
    """
    论文结果类
    功能：存储单篇论文的基本信息和解析后的内容
    属性：
        title: 论文标题
        abstract: 论文摘要
        article: 解析后的论文全文内容（字典格式）
        citations_count: 引用次数
        year: 发表年份
    """
    def __init__(
        self, title="", abstract="", article=None, citations_count=0, year=None
    ) -> None:
        self.title = title  # 论文标题
        self.abstract = abstract  # 论文摘要
        self.article = article  # 解析后的论文全文内容，通常是一个包含sections、references等信息的字典
        self.citations_count = citations_count  # 该论文被引用的次数
        self.year = year  # 论文发表年份