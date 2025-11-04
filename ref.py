# 导入必要的库
import requests  # 用于发送HTTP请求，主要用于同步请求
import scipdf  # 用于解析PDF文件，提取学术论文的结构化内容
import os  # 用于文件路径操作和系统环境变量
import aiohttp  # 用于异步HTTP请求，提高并发性能
import asyncio  # 用于异步编程，处理并发任务
import numpy as np  # 用于数值计算，特别是向量相似度计算
import random  # 用于随机采样
import logging  # 用于日志记录

# 从环境变量中获取Semantic Scholar API密钥，用于提高API调用限制
api_key = os.environ.get("SEMANTIC_SEARCH_API_KEY")

# 创建日志记录器，用于记录程序运行状态和错误信息
logger = logging.getLogger(__name__)


def get_content_between_a_b(start_tag, end_tag, text):
    """
    从文本中提取指定标签之间的内容
    功能：在文本中查找开始标签和结束标签，提取它们之间的所有内容
    参数：
        start_tag: 开始标签，如"<title>"
        end_tag: 结束标签，如"</title>"
        text: 要搜索的文本内容
    返回：提取出的文本内容，去除首尾空白
    """
    extracted_text = ""  # 存储提取的文本内容
    start_index = text.find(start_tag)  # 查找第一个开始标签的位置
    while start_index != -1:  # 循环查找所有匹配的标签对
        end_index = text.find(end_tag, start_index + len(start_tag))  # 查找对应的结束标签
        if end_index != -1:  # 如果找到结束标签
            # 提取标签之间的内容并添加到结果中
            extracted_text += text[start_index + len(start_tag) : end_index] + " "
            # 继续查找下一个开始标签
            start_index = text.find(start_tag, end_index + len(end_tag))
        else:
            break  # 如果没找到结束标签，退出循环

    return extracted_text.strip()  # 返回去除首尾空白的提取内容


def extract(text, type):
    """
    从文本中提取指定类型的内容
    功能：根据类型参数构造标签，提取对应内容，如果提取失败则返回原文本
    参数：
        text: 要处理的文本内容
        type: 要提取的内容类型，如"title"、"abstract"等
    返回：提取的内容或原文本
    """
    if text:  # 如果输入文本不为空
        # 构造标签格式，如"<title>"和"</title>"
        target_str = get_content_between_a_b(f"<{type}>", f"</{type}>", text)
        if target_str:  # 如果成功提取到内容
            return target_str
        else:  # 如果提取失败，返回原文本
            return text
    else:  # 如果输入文本为空，返回空字符串
        return ""


async def fetch(url):
    """
    异步获取URL内容
    功能：使用aiohttp异步获取网页内容，模拟浏览器请求
    参数：
        url: 要获取的URL地址
    返回：网页内容的字节数据，失败时返回None
    """
    try:
        # 设置120秒的超时时间
        timeout = aiohttp.ClientTimeout(total=120)
        # 设置User-Agent头，模拟Chrome浏览器，避免被反爬虫机制拦截
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/87.0.4280.88 Safari/537.36"
        }  # 模拟常见浏览器的User-Agent
        # 创建异步HTTP会话
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # 发送GET请求，允许重定向
            async with session.get(
                url, headers=headers, allow_redirects=True
            ) as response:
                if response.status == 200:  # 如果请求成功
                    content = await response.read()  # 读取响应内容
                    logger.info(f"Successfully fetched the URL: {url}")
                    return content
                else:  # 如果请求失败
                    logger.error(
                        f"Failed to fetch the URL: {url} with status code: {response.status}"
                    )
                    return None
    except Exception as e:  # 捕获所有异常
        logger.error(
            f"An unexpected error occurred while fetching the URL: {url}: {str(e)}"
        )
        return None


class Result:
    """
    论文结果类
    功能：存储单篇论文的基本信息和解析后的内容
    属性：
        title: 论文标题
        abstract: 论文摘要
        article: 解析后的论文全文内容（字典格式）
        citations_conut: 引用次数
        year: 发表年份
    """
    def __init__(
        self, title="", abstract="", article=None, citations_conut=0, year=None
    ) -> None:
        self.title = title  # 论文标题
        self.abstract = abstract  # 论文摘要
        self.article = article  # 解析后的论文全文内容，通常是一个包含sections、references等信息的字典
        self.citations_conut = citations_conut  # 该论文被引用的次数
        self.year = year  # 论文发表年份


# 定义Semantic Scholar API的字段列表
# 这些字段用于指定从API返回的论文信息中包含哪些数据

semantic_fields = [
    "title",  # 论文标题
    "abstract",  # 论文摘要
    "year",  # 发表年份
    "authors.name",  # 作者姓名
    "authors.paperCount",  # 作者发表的论文数量
    "authors.citationCount",  # 作者的总引用次数
    "authors.hIndex",  # 作者的h指数
    "url",  # 论文URL
    "referenceCount",  # 参考文献数量
    "citationCount",  # 引用次数
    "influentialCitationCount",  # 高影响力引用次数
    "isOpenAccess",  # 是否为开放获取
    "openAccessPdf",  # 开放获取PDF链接
    "fieldsOfStudy",  # 研究领域
    "s2FieldsOfStudy",  # Semantic Scholar研究领域分类
    "embedding.specter_v1",  # SPECTER v1嵌入向量
    "embedding.specter_v2",  # SPECTER v2嵌入向量
    "publicationDate",  # 发表日期
    "citations",  # 引用该论文的其他论文信息
]


# 定义Semantic Scholar支持的研究领域列表
# 这些领域可以用于过滤搜索结果，只返回特定领域的论文

fieldsOfStudy = [
    "Computer Science",  # 计算机科学
    "Medicine",  # 医学
    "Chemistry",  # 化学
    "Biology",  # 生物学
    "Materials Science",  # 材料科学
    "Physics",  # 物理学
    "Geology",  # 地质学
    "Art",  # 艺术
    "History",  # 历史学
    "Geography",  # 地理学
    "Sociology",  # 社会学
    "Business",  # 商学
    "Political Science",  # 政治学
    "Philosophy",  # 哲学
    "Art",  # 艺术（重复项）
    "Literature",  # 文学
    "Music",  # 音乐
    "Economics",  # 经济学
    "Mathematics",  # 数学
    "Engineering",  # 工程学
    "Environmental Science",  # 环境科学
    "Agricultural and Food Sciences",  # 农业和食品科学
    "Education",  # 教育学
    "Law",  # 法学
    "Linguistics",  # 语言学
]

# 注释：citations字段可以包含的子字段列表
# citations.paperId, citations.title, citations.year, citations.authors.name, citations.authors.paperCount, citations.authors.citationCount, citations.authors.hIndex, citations.url, citations.referenceCount, citations.citationCount, citations.influentialCitationCount, citations.isOpenAccess, citations.openAccessPdf, citations.fieldsOfStudy, citations.s2FieldsOfStudy, citations.publicationDate

# 注释：publicationDateOrYear参数的格式示例
# publicationDateOrYear： 2019-03-05 ； 2019-03 ； 2019 ； 2016-03-05:2020-06-06 ； 1981-08-25: ； :2020-06-06 ； 1981:2020

# 注释：publicationTypes参数支持的类型
# publicationTypes: Review ； JournalArticle CaseReport ； ClinicalTrial ； Dataset ； Editorial ； LettersAndComments ； MetaAnalysis ； News ； Study ； Book ； BookSection


def process_fields(fields):
    """
    处理字段列表
    功能：将字段列表转换为逗号分隔的字符串，用于API请求
    参数：
        fields: 字段列表
    返回：逗号分隔的字段字符串
    """
    return ",".join(fields)  # 将列表中的字段用逗号连接成字符串


class SemanticSearcher:
    """
    语义搜索器类
    功能：使用Semantic Scholar API进行学术论文搜索、下载和解析
    主要功能：
        1. 根据查询词搜索相关论文
        2. 下载开放获取的PDF论文
        3. 解析PDF论文内容
        4. 计算论文间的语义相似度
        5. 重新排序搜索结果
        6. 查找相关论文（引用和参考文献）
    """
    def __init__(self, save_file="papers/", ban_paper=[]) -> None:
        """
        初始化语义搜索器
        参数：
            save_file: 论文PDF文件的保存目录，默认为"papers/"
            ban_paper: 需要排除的论文标题列表，默认为空列表
        """
        self.save_file = save_file  # 设置论文保存目录
        self.ban_paper = ban_paper  # 设置需要排除的论文列表

    async def search_papers_async(
        self,
        query,
        limit=10,
        offset=0,
        fields=[
            "title",
            "paperId",
            "abstract",
            "isOpenAccess",
            "openAccessPdf",
            "year",
            "publicationDate",
            "citations.title",
            "citations.abstract",
            "citations.isOpenAccess",
            "citations.openAccessPdf",
            "citations.citationCount",
            "citationCount",
            "citations.year",
        ],
        publicationDate=None,
        minCitationCount=0,
        year=None,
        publicationTypes=None,
        fieldsOfStudy=None,
    ):
        """
        异步搜索论文
        功能：调用Semantic Scholar API搜索学术论文
        参数：
            query: 搜索查询词
            limit: 返回结果数量限制，默认10
            offset: 结果偏移量，用于分页，默认0
            fields: 要返回的字段列表，包含论文基本信息和引用信息
            publicationDate: 发表日期过滤，格式如"2019-03-05"或"2019"
            minCitationCount: 最小引用次数过滤，默认0
            year: 年份过滤，默认None
            publicationTypes: 出版物类型过滤，如"JournalArticle"
            fieldsOfStudy: 研究领域过滤，如"Computer Science"
        返回：API返回的JSON数据，包含搜索结果
        """
        url = "https://api.semanticscholar.org/graph/v1/paper/search"  # Semantic Scholar API搜索端点
        # 如果fields是列表，则转换为逗号分隔的字符串
        fields = process_fields(fields) if isinstance(fields, list) else fields

        # 构建查询参数字典
        query_params = {
            "query": query,  # 搜索查询词
            "limit": limit,  # 结果数量限制
            "offset": offset,  # 结果偏移量
            "fields": fields,  # 要返回的字段
            "publicationDateOrYear": publicationDate,  # 发表日期过滤
            "minCitationCount": minCitationCount,  # 最小引用次数
            "year": year,  # 年份过滤
            "publicationTypes": publicationTypes,  # 出版物类型过滤
            "fieldsOfStudy": fieldsOfStudy,  # 研究领域过滤
        }

        await asyncio.sleep(0.5)  # 添加延迟，避免API请求过于频繁

        try:
            # 过滤掉None值的参数，只保留有效的查询参数
            filtered_query_params = {
                key: value for key, value in query_params.items() if value is not None
            }

            # 如果有API密钥，则添加到请求头中
            headers = {"x-api-key": api_key} if api_key else None
            # 发送GET请求到Semantic Scholar API
            response = requests.get(url, params=filtered_query_params, headers=headers)

            if response.status_code == 200:  # 请求成功
                response_data = response.json()  # 解析JSON响应
                logger.info(f"Search successful for query: {query}")
                return response_data
            elif response.status_code == 429:  # 请求过于频繁，需要重试
                await asyncio.sleep(5)  # 等待5秒后重试
                logger.warning(
                    f"Request failed with status code {response.status_code}: begin to retry"
                )
                # 递归调用自身进行重试
                return await self.search_papers_async(
                    query,
                    limit,
                    offset,
                    fields,
                    publicationDate,
                    minCitationCount,
                    year,
                    publicationTypes,
                    fieldsOfStudy,
                )
            else:  # 其他错误状态码
                logger.error(
                    f"Request failed with status code {response.status_code}: {response.text}"
                )
                return None
        except requests.RequestException as e:  # 捕获网络请求异常
            logger.error(f"An error occurred: {e}")
            return None

    def cal_cosine_similarity(self, vec1, vec2):
        """
        计算两个向量的余弦相似度
        功能：计算两个向量之间的余弦相似度，用于衡量语义相似性
        参数：
            vec1: 第一个向量
            vec2: 第二个向量
        返回：余弦相似度值（-1到1之间）
        公式：cos(θ) = (A·B) / (||A|| * ||B||)
        """
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def cal_cosine_similarity_matric(self, matric1, matric2):
        """
        计算两个矩阵之间的余弦相似度矩阵
        功能：批量计算多个向量之间的余弦相似度，用于论文重排序
        参数：
            matric1: 第一个矩阵（查询向量）
            matric2: 第二个矩阵（论文向量）
        返回：相似度分数列表
        """
        # 确保输入是numpy数组
        if isinstance(matric1, list):
            matric1 = np.array(matric1)
        if isinstance(matric2, list):
            matric2 = np.array(matric2)
        # 确保矩阵是二维的，如果是向量则转换为行向量
        if len(matric1.shape) == 1:
            matric1 = matric1.reshape(1, -1)
        if len(matric2.shape) == 1:
            matric2 = matric2.reshape(1, -1)
        
        # 计算矩阵乘法：matric1 @ matric2.T
        dot_product = np.dot(matric1, matric2.T)
        # 计算每个向量的L2范数
        norm1 = np.linalg.norm(matric1, axis=1)
        norm2 = np.linalg.norm(matric2, axis=1)

        # 计算余弦相似度：点积除以范数的外积
        cos_sim = dot_product / np.outer(norm1, norm2)
        scores = cos_sim.flatten()  # 将结果展平为一维数组
        return scores.tolist()  # 转换为Python列表返回

    def rerank_papers(self, query_embedding, paper_list, llm):
        """
        根据语义相似度重新排序论文列表
        功能：使用查询向量和论文内容的嵌入向量计算相似度，重新排序论文
        参数：
            query_embedding: 查询的嵌入向量
            paper_list: 论文列表
            llm: 语言模型对象，用于生成嵌入向量
        返回：按相似度排序的论文列表
        """
        if len(paper_list) == 0:  # 如果论文列表为空，直接返回空列表
            return []
        # 过滤掉空值论文
        paper_list = [paper for paper in paper_list if paper]
        paper_contents = []  # 存储论文内容文本
        # 为每篇论文构建包含标题和摘要的文本内容
        for paper in paper_list:
            paper_content = f"Title: {paper['title']}\nAbstract: {paper['abstract']}"
            paper_contents.append(paper_content)
        
        # 使用语言模型获取论文内容的嵌入向量
        paper_contents_embbeding = llm.get_embbeding(paper_contents)
        paper_contents_embbeding = np.array(paper_contents_embbeding)
        
        # 计算查询向量与论文内容向量的余弦相似度
        scores = self.cal_cosine_similarity_matric(
            query_embedding, paper_contents_embbeding
        )

        # 根据相似度分数对论文列表进行排序（降序）
        paper_list = sorted(zip(paper_list, scores), key=lambda x: x[1], reverse=True)
        # 提取排序后的论文（去掉分数）
        paper_list = [paper[0] for paper in paper_list]
        return paper_list

    async def search_async(
        self,
        query,
        max_results=5,
        paper_list=None,
        rerank_query=None,
        llm=None,
        year=None,
        publicationDate=None,
        need_download=True,
        fields=[
            "title",
            "paperId",
            "abstract",
            "isOpenAccess",
            "openAccessPdf",
            "year",
            "publicationDate",
            "citationCount",
        ],
    ):
        """
        异步搜索论文的主要方法
        功能：搜索论文、过滤结果、下载PDF、解析内容并返回结构化结果
        参数：
            query: 搜索查询词
            max_results: 最大返回结果数，默认5
            paper_list: 已读论文列表，用于避免重复
            rerank_query: 用于重排序的查询词，如果提供则进行语义重排序
            llm: 语言模型对象，用于生成嵌入向量和重排序
            year: 年份过滤
            publicationDate: 发表日期过滤
            need_download: 是否需要下载PDF文件，默认True
            fields: 要返回的字段列表
        返回：Result对象列表，包含论文的完整信息
        """

        # 处理已读论文列表，避免重复搜索
        readed_papers = []  # 存储已读论文标题
        if paper_list:  # 如果提供了已读论文列表
            if isinstance(paper_list, set):  # 如果是集合，转换为列表
                paper_list = list(paper_list)
            if len(paper_list) == 0:  # 如果列表为空，跳过
                pass
            elif isinstance(paper_list[0], str):  # 如果列表包含字符串（论文标题）
                readed_papers = paper_list
            elif isinstance(paper_list[0], Result):  # 如果列表包含Result对象
                readed_papers = [paper.title for paper in paper_list]  # 提取标题

        logger.info(f"Searching for papers related to query : <{query}>")

        # 设置搜索限制，通常搜索更多结果以便后续过滤
        nlimit = max_results * 6
        # 调用API搜索论文
        results = await self.search_papers_async(
            query,
            limit=nlimit,
            year=year,
            publicationDate=publicationDate,
            fields=fields,
        )

        # 检查搜索结果是否有效
        if not results or "data" not in results:
            return []

        # 过滤掉被禁止的论文
        new_results = []
        for result in results["data"]:
            if result["title"] in self.ban_paper:  # 如果论文在禁止列表中，跳过
                continue
            new_results.append(result)
        results = new_results

        # 根据是否需要下载PDF来筛选论文候选
        if need_download:
            paper_candidates = []  # 存储候选论文
            for result in results:
                # 如果PDF文件已存在且论文未被读过，直接加入候选列表
                if (
                    os.path.exists(
                        os.path.join(self.save_file, f"{result['title']}.pdf")
                    )
                    and result["title"] not in readed_papers
                ):
                    paper_candidates.append(result)
                # 如果论文不是开放获取或没有PDF链接，跳过
                elif not result["isOpenAccess"] or not result["openAccessPdf"]:
                    continue
                else:  # 其他情况（开放获取且有PDF链接），加入候选列表
                    paper_candidates.append(result)
        else:  # 如果不需要下载，直接使用所有结果
            paper_candidates = results

        # 如果提供了语言模型和重排序查询，进行语义重排序
        if llm and rerank_query:
            # 获取重排序查询的嵌入向量
            rerank_query_embbeding = llm.get_embbeding(rerank_query)
            rerank_query_embbeding = np.array(rerank_query_embbeding)
            # 根据语义相似度重新排序论文
            paper_candidates = self.rerank_papers(
                rerank_query_embbeding, paper_candidates, llm
            )

        # 处理最终结果，创建Result对象
        final_results = []
        for result in paper_candidates:
            article = None  # 初始化论文内容
            if need_download:  # 如果需要下载PDF
                # 如果PDF文件已存在，直接从本地读取
                if os.path.exists(
                    os.path.join(self.save_file, f"{result['title']}.pdf")
                ):
                    article = self.read_arxiv_from_path(
                        os.path.join(self.save_file, f"{result['title']}.pdf")
                    )
                else:  # 如果PDF文件不存在，从网络下载
                    pdf_link = result["openAccessPdf"]["url"]
                    article = await self.read_arxiv_from_link_async(
                        pdf_link, f"{result['title']}.pdf"
                    )
                if not article:  # 如果解析失败，跳过该论文
                    continue
            # 提取论文基本信息
            title, abstract, citationCount, year = (
                result["title"],
                result["abstract"],
                result["citationCount"],
                result["year"],
            )
            # 创建Result对象并添加到最终结果列表
            final_results.append(Result(title, abstract, article, citationCount, year))
            # 如果达到最大结果数，停止处理
            if len(final_results) >= max_results:
                break
        return final_results

    async def search_related_paper_async(
        self,
        title,
        need_citation=True,
        need_reference=True,
        rerank_query=None,
        llm=None,
        paper_list=[],
    ):
        """
        异步搜索相关论文（引用和参考文献）
        功能：根据给定论文标题，查找引用该论文的论文和该论文引用的论文
        参数：
            title: 目标论文标题
            need_citation: 是否需要查找引用该论文的论文，默认True
            need_reference: 是否需要查找该论文引用的论文，默认True
            rerank_query: 用于重排序的查询词
            llm: 语言模型对象，用于重排序
            paper_list: 已读论文列表，用于避免重复
        返回：相关论文的Result对象，如果失败返回None
        """

        logger.info(
            f"Searching for related papers of paper <{title}>; Citation:{need_citation}; Reference:{need_reference}"
        )

        fileds = [
            "title",
            "abstract",
            "citations.title",
            "citations.abstract",
            "citations.citationCount",
            "references.title",
            "references.abstract",
            "references.citationCount",
            "citations.isOpenAccess",
            "citations.openAccessPdf",
            "references.isOpenAccess",
            "references.openAccessPdf",
            "citations.year",
            "references.year",
        ]
        results = await self.search_papers_async(title, limit=3, fields=fileds)

        related_papers = []
        related_papers_title = []
        if not results or "data" not in results:
            logger.warning(
                f"Failed to find related papers of paper <{title}>; Citation:{need_citation}; Reference:{need_reference}"
            )
            return None
        for result in results["data"]:
            if not result:
                continue
            if need_citation:
                for citation in result["citations"]:
                    if (
                        os.path.exists(
                            os.path.join(self.save_file, f"{citation['title']}.pdf")
                        )
                        and citation["title"] not in paper_list
                    ):
                        if (
                            "openAccessPdf" not in citation
                            or not citation["openAccessPdf"]
                            or "url" not in citation["openAccessPdf"]
                        ):
                            citation["openAccessPdf"] = {"url": None}
                        related_papers.append(citation)
                        related_papers_title.append(citation["title"])
                    elif (
                        citation["title"] in related_papers_title
                        or citation["title"] in self.ban_paper
                        or citation["title"] in paper_list
                    ):
                        continue
                    elif (
                        citation["isOpenAccess"] == False
                        or citation["openAccessPdf"] == None
                    ):
                        continue
                    else:
                        related_papers.append(citation)
                        related_papers_title.append(citation["title"])
            if need_reference and result["references"]:
                for reference in result["references"]:
                    if (
                        os.path.exists(
                            os.path.join(self.save_file, f"{reference['title']}.pdf")
                        )
                        and reference["title"] not in paper_list
                    ):
                        if (
                            "openAccessPdf" not in reference
                            or not reference["openAccessPdf"]
                            or "url" not in reference["openAccessPdf"]
                        ):
                            reference["openAccessPdf"] = {"url": None}
                        related_papers.append(reference)
                        related_papers_title.append(reference["title"])
                    elif (
                        reference["title"] in related_papers_title
                        or reference["title"] in self.ban_paper
                        or reference["title"] in paper_list
                    ):
                        continue
                    elif (
                        reference["isOpenAccess"] == False
                        or reference["openAccessPdf"] == None
                    ):
                        continue
                    else:
                        related_papers.append(reference)
                        related_papers_title.append(reference["title"])
            if result:
                break

        if len(related_papers) >= 200:
            related_papers = random.sample(related_papers, 200)

        if rerank_query and llm:
            rerank_query_embbeding = llm.get_embbeding(rerank_query)
            rerank_query_embbeding = np.array(rerank_query_embbeding)
            related_papers = self.rerank_papers(
                rerank_query_embbeding, related_papers, llm
            )
            related_papers = [
                [
                    paper["title"],
                    paper["abstract"],
                    paper["openAccessPdf"]["url"],
                    paper["citationCount"],
                    paper["year"],
                ]
                for paper in related_papers
            ]
        else:
            related_papers = [
                [
                    paper["title"],
                    paper["abstract"],
                    paper["openAccessPdf"]["url"],
                    paper["citationCount"],
                    paper["year"],
                ]
                for paper in related_papers
            ]
            related_papers = sorted(related_papers, key=lambda x: x[3], reverse=True)
        logger.info(f"Found {len(related_papers)} related papers")
        for paper in related_papers:
            url = paper[2]
            article = await self.read_arxiv_from_link_async(url, f"{paper[0]}.pdf")
            if not article:
                continue
            result = Result(paper[0], paper[1], article, paper[3], paper[4])
            logger.info(f"Successfully found related papers of paper <{title}>")
            return result
        logger.warning(
            f"Failed to find related papers of paper <{title}>; Citation:{need_citation}; Reference:{need_reference}"
        )
        return None

    async def read_arxiv_from_link_async(self, pdf_link, filename):
        """
        异步从PDF链接读取论文内容
        功能：检查本地是否已有PDF文件，如果没有则下载并解析
        参数：
            pdf_link: PDF文件的下载链接
            filename: 保存的文件名
        返回：解析后的论文内容字典，失败时返回None
        """
        file_path = os.path.join(self.save_file, filename)  # 构建本地文件路径
        # 如果文件已存在，直接从本地读取
        if os.path.exists(file_path):
            article_dict = self.read_arxiv_from_path(file_path)
            return article_dict

        # 如果文件不存在，先下载PDF
        result = await self.download_pdf_async(pdf_link, file_path)
        if not result:  # 如果下载失败
            logger.error(f"Failed to download the PDF file: {filename}")
            return None
        try:  # 尝试解析下载的PDF文件
            article_dict = self.read_arxiv_from_path(file_path)
            return article_dict
        except Exception as e:  # 如果解析失败
            logger.error(
                f"Failed to read the article from the PDF file: {e}, {filename}"
            )
            return None

    def read_arxiv_from_path(self, pdf_path):
        """
        从本地PDF文件路径读取论文内容
        功能：使用scipdf库解析PDF文件，提取结构化内容
        参数：
            pdf_path: PDF文件的本地路径
        返回：解析后的论文内容字典，包含标题、摘要、章节、参考文献等
        """
        if not os.path.exists(pdf_path):  # 检查文件是否存在
            logger.error(f"The PDF file <{pdf_path}> does not exist.")
            return None
        try:
            # 使用scipdf库解析PDF文件为字典格式
            article_dict = scipdf.parse_pdf_to_dict(pdf_path)
            logger.info(f"Successfully parsed the PDF file: {article_dict}")
        except Exception as e:  # 如果解析失败
            return None
        return article_dict

    # async def download_pdf_async(self, pdf_link, save_path):
    #     if os.path.exists(save_path):
    #         logger.info(f"The PDF file <{save_path}> already exists.")
    #         return True
    #     content = await fetch(pdf_link)
    #     if not content:
    #         logger.error(f"Failed to download the PDF: {save_path}")
    #         return False
    #     try:
    #         with open(save_path, 'wb') as file:
    #             file.write(content)
    #         logger.info(f"Successfully downloaded the PDF file: {save_path}")
    #         return True
    #     except Exception as e:
    #         logger.error(f"Failed to download the PDF file: {e}, {save_path}")
    #         return False

    async def download_pdf_async(
        self, pdf_link, save_path, user_agent: str = "requests/2.0.0"
    ):
        """
        异步下载PDF文件
        功能：从网络下载PDF文件并保存到本地
        参数：
            pdf_link: PDF文件的下载链接
            save_path: 本地保存路径
            user_agent: 用户代理字符串，默认"requests/2.0.0"
        返回：下载成功返回True，失败返回False
        """
        # 如果文件已存在，直接返回成功
        if os.path.exists(save_path):
            logger.info(f"The file <{save_path}> already exists.")
            return True

        try:
            # 确保目标目录存在
            dirpath = os.path.dirname(save_path)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)

            # 设置120秒的超时时间
            timeout = aiohttp.ClientTimeout(total=120)
            # 设置请求头
            headers = {
                "user-agent": user_agent,
            }

            # 创建异步HTTP会话
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # 发送GET请求下载文件
                async with session.get(
                    pdf_link, headers=headers, allow_redirects=True, ssl=False
                ) as response:
                    # 检查请求是否成功
                    response.raise_for_status()

                    # 获取内容类型
                    content_type = (response.headers.get("content-type") or "").lower()

                    # 根据内容类型决定保存路径
                    if content_type.startswith("application/pdf"):  # 如果是PDF文件
                        target_path = save_path
                    # elif content_type.startswith('text/html'):  # 注释掉的HTML处理
                    #     root, _ = os.path.splitext(save_path)
                    #     target_path = root + '.html'
                    #     logger.info(f"Content-Type is HTML; saving to {target_path}")
                    else:  # 不支持的内容类型
                        logger.error(f"Unsupported Content-Type: {content_type}")
                        return False

                    # 分块流式写入文件
                    with open(target_path, "wb") as f:
                        async for chunk in response.content.iter_chunked(8192):  # 8KB块大小
                            f.write(chunk)

            logger.info(f"Successfully saved file to: {target_path}")
            return True
        except Exception as e:  # 捕获所有异常
            logger.error(f"Failed to download the file: {e}, {save_path}")
            return False

    def read_paper_title_abstract(self, article):
        """
        读取论文的标题和摘要
        功能：从解析后的论文内容中提取标题和摘要
        参数：
            article: 解析后的论文内容字典
        返回：格式化的标题和摘要文本
        """
        title = article["title"]  # 提取标题
        abstract = article["abstract"]  # 提取摘要
        paper_content = f"""
            Title: {title}
            Abstract: {abstract}
        """
        return paper_content

    def read_paper_title_abstract_introduction(self, article):
        """
        读取论文的标题、摘要和引言
        功能：从解析后的论文内容中提取标题、摘要和引言部分
        参数：
            article: 解析后的论文内容字典
        返回：格式化的标题、摘要和引言文本
        """
        title = article["title"]  # 提取标题
        abstract = article["abstract"]  # 提取摘要
        introduction = article["sections"][0]["text"]  # 提取第一个章节（通常是引言）的文本
        paper_content = f"""
            Title: {title}
            Abstract: {abstract}
            Introduction: {introduction}
        """
        return paper_content

    def read_paper_content(self, article):
        """
        读取论文的完整内容
        功能：从解析后的论文内容中提取标题、摘要和所有章节
        参数：
            article: 解析后的论文内容字典
        返回：格式化的完整论文内容文本
        """
        paper_content = self.read_paper_title_abstract(article)  # 先获取标题和摘要
        # 遍历所有章节，添加章节标题、内容和参考文献ID
        for section in article["sections"]:
            paper_content += f"section: {section['heading']}\n content: {section['text']}\n ref_ids: {section['publication_ref']}\n"
        return paper_content

    def read_paper_content_with_ref(self, article):
        """
        读取论文的完整内容和参考文献
        功能：从解析后的论文内容中提取所有信息，包括参考文献列表
        参数：
            article: 解析后的论文内容字典
        返回：格式化的完整论文内容文本，包含参考文献
        """
        paper_content = self.read_paper_content(article)  # 先获取完整内容
        paper_content += "<References>\n"  # 添加参考文献开始标记
        # 遍历所有参考文献，添加引用ID、标题和年份
        for refer in article["references"]:
            ref_id = refer["ref_id"]  # 参考文献ID
            title = refer["title"]  # 参考文献标题
            year = refer["year"]  # 参考文献年份
            paper_content += f"Ref_id:{ref_id} Title: {title} Year: ({year})\n"
        paper_content += "</References>\n"  # 添加参考文献结束标记
        return paper_content
