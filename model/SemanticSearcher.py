# model/SemanticSearcher.py
# 导入外部库
from logging import Logger
from numpy import ndarray
from typing import List, Any
import numpy as np
import requests
import asyncio
import scipdf
import random
import aiohttp
import logging
import time
import os


# 导入内部库
from model.Result import Result


logger = logging.getLogger(__name__)


class SemanticSearcher:
    def __init__(self, save_dir: str = "papers/", ban_list: List[str] = []) -> None:
        self.save_dir = save_dir
        self.ban_list = ban_list
    
    # 异步调用 Semantic Scholar API 搜索论文
    async def search_papers_async(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        fields: List[str] = [
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
        publicationDate: str | None = None,
        minCitationCount: int = 0,
        year: int | None = None,
        publicationTypes: List[str] | None = None,
        fieldsOfStudy: List[str] | None = None,
        api_key: str | None = None,
    ) -> dict | None:
        """
        异步搜索论文
        功能：调用Semantic Scholar API搜索学术论文
        参数：
            query: 搜索查询词
            limit: 返回结果数量限制，默认10
            offset: 结果偏移量，用于分页，默认0
            fields: 要返回的字段列表，包含论文基本信息和引用信息
            publicationDate: 发表日期过滤，格式如"2019-03-05"或"2019"
        """
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        fields = ",".join(fields) if isinstance(fields, list) else fields
        
        query_params = {
            "query": query,
            "limit": limit,
            "offset": offset,
            "fields": fields,
            "publicationDateOrYear": publicationDate,
            "minCitationCount": minCitationCount,
            "year": year,
            "publicationTypes": publicationTypes,
            "fieldsOfStudy": fieldsOfStudy,
        }
        
        await asyncio.sleep(0.5)
        
        try:
            filtered_query_params = {
                key: value for key, value in query_params.items() if value is not None
            }
            
            headers = {"x-api-key": api_key} if api_key else None
            
            response = requests.get(url, params=filtered_query_params, headers=headers)
            
            if response.status_code == 200:
                response_data = response.json()
                logger.info(f"Search successful for query: {query}")
                return response_data
            
            elif response.status_code == 429:
                await asyncio.sleep(5)
                logger.warning(f"Request failed with status code {response.status_code}: begin to retry")
                
                return await self.search_papers_async(query, limit, offset, fields, publicationDate, minCitationCount, year, publicationTypes, fieldsOfStudy, api_key)
            else:
                logger.error(f"Request failed with status code {response.status_code}: {response.text}")
                return None
        except requests.RequestException as e:
            logger.error(f"An error occurred: {e}")
            return None
    
    # 计算两个向量的余弦相似度
    def cal_cosine_similarity(self, vec1: ndarray, vec2: ndarray) -> float:
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
    
    # 计算两个矩阵之间的余弦相似度矩阵
    def _cal_cosine_similarity_matric(self, matric1: ndarray, matric2: ndarray) -> List[float]:
        """
        计算两个矩阵之间的余弦相似度矩阵
        功能：批量计算多个向量之间的余弦相似度，用于论文重排序
        参数：
            matric1: 第一个矩阵（查询向量）
            matric2: 第二个矩阵（论文向量）
        返回：相似度分数列表
        """
        if isinstance(matric1, list):
            matric1 = np.array(matric1)
        if isinstance(matric2, list):
            matric2 = np.array(matric2)
        
        if len(matric1.shape) == 1:
            matric1 = matric1.reshape(1, -1)
        if len(matric2.shape) == 1:
            matric2 = matric2.reshape(1, -1)
        
        dot_product = np.dot(matric1, matric2.T)
        norm1 = np.linalg.norm(matric1, axis=1)
        norm2 = np.linalg.norm(matric2, axis=1)
        
        cos_sim = dot_product / np.outer(norm1, norm2)
        scores = cos_sim.flatten()
        return scores.tolist()
    
    # 根据语义相似度重新排序论文列表
    def rerank_papers(self, query_embedding: ndarray, paper_list: List[dict], llm) -> List[dict]:
        """
        根据语义相似度重新排序论文列表
        功能：使用查询向量和论文内容的嵌入向量计算相似度，重新排序论文
        参数：
            query_embedding: 查询的嵌入向量
            paper_list: 论文列表
            llm: 语言模型对象，用于生成嵌入向量
        返回：按相似度排序的论文列表
        """
        if len(paper_list) == 0:
            return []
        
        paper_list = [paper for paper in paper_list if paper]
        paper_contents = []
        for paper in paper_list:
            paper_content = f"Title: {paper['title']}\nAbstract: {paper['abstract']}"
            paper_contents.append(paper_content)
        
        paper_contents_embedding = llm.get_embedding(paper_contents)
        paper_contents_embedding = np.array(paper_contents_embedding)
        
        scores = self._cal_cosine_similarity_matric(
            query_embedding, paper_contents_embedding
        )
        
        paper_list = sorted(zip[tuple[Any, Any]](paper_list, scores), key=lambda x: x[1], reverse=True)
        
        paper_list = [paper[0] for paper in paper_list]
        return paper_list
        
    # 从本地PDF路径读取论文内容
    async def read_arxiv_from_path(self, pdf_path: str) -> Result | None:
        """
        异步从本地PDF文件读取论文内容
        功能：从本地PDF文件读取论文内容
        参数：
            pdf_path: 本地PDF文件路径
            logger: 日志记录器
        返回：解析后的论文内容字典，失败时返回None
        """
        if not os.path.exists(pdf_path):
            logger.error(f"The PDF file <{pdf_path}> does not exist.")
            return None
        try:
            article_dict = scipdf.parse_pdf_to_dict(pdf_path)
            logger.info(f"Successfully parsed the PDF file: {article_dict}")
            return article_dict
        except Exception as e:
            logger.error(f"Failed to read the article from the PDF file: {e}, {pdf_path}")
            return None
    
    # 提取论文标题和摘要
    def read_paper_title_abstract(self, article: dict) -> tuple[str, str]:
        """
        从论文内容中提取标题和摘要
        功能：从论文内容中提取标题和摘要
        参数：
            article: 论文内容字典
        返回：标题和摘要
        """
        title = article["title"]  # 提取标题
        abstract = article["abstract"]  # 提取摘要
        paper_content = f"""
            Title: {title}
            Abstract: {abstract}
        """
        return paper_content
    
    # 提取标题、摘要和引言
    def read_paper_title_abstract_introduction(self, article: dict) -> tuple[str, str]:
        """
        从论文内容中提取标题、摘要和引言
        功能：从论文内容中提取标题、摘要和引言
        参数：
            article: 论文内容字典
        返回：标题、摘要和引言
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
    
    # 提取论文完整内容
    def read_paper_content(self, article: dict) -> str:
        """
        从论文内容中提取完整内容
        功能：从论文内容中提取完整内容
        参数：
            article: 论文内容字典
        返回：完整内容
        """
        paper_content = self.read_paper_title_abstract(article)  # 先获取标题和摘要
        # 遍历所有章节，添加章节标题、内容和参考文献ID
        for section in article["sections"]:
            paper_content += f"section: {section['heading']}\n content: {section['text']}\n ref_ids: {section['publication_ref']}\n"
        return paper_content
    
    # 提取完整内容和参考文献
    def read_paper_content_with_ref(self, article: dict) -> str:
        """
        从论文内容中提取完整内容和参考文献
        功能：从论文内容中提取完整内容和参考文献
        参数：
            article: 论文内容字典
        返回：完整内容和参考文献
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
    
    
    
    # 主搜索方法，整合搜索、过滤、下载、解析流程
    async def search_async(
        self,
        query: str,
        max_results: int = 5,
        paper_list: List[Result] | None = None,
        rerank_query: str | None = None,
        llm: Any | None = None,
        year: int | None = None,
        publicationDate: str | None = None,
        need_download: bool = True,
        fields: List[str] = [
            "title",
            "paperId",
            "abstract",
            "isOpenAccess",
            "openAccessPdf",
            "year",
            "publicationDate",
            "citationCount",
        ],
        api_key: str | None = None,
    ) -> List[Result] | None:
        """
        异步搜索论文
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
            api_key: Semantic Scholar API密钥
        """
        readed_papers = []
        if paper_list:
            if isinstance(paper_list, set):
                paper_list = list(paper_list)
            if len(paper_list) == 0:
                pass
            elif isinstance(paper_list[0], str):
                readed_papers = paper_list
            elif isinstance(paper_list[0], Result):
                readed_papers = [paper.title for paper in paper_list]
        
        logger.info(f"Searching for papers related to query : <{query}>")
        # 搜寻论文
        nlimit = max_results * 6
        
        start_time = time.time()
        results = await self.search_papers_async(
            query,
            limit=nlimit,
            year=year,
            publicationDate=publicationDate,
            fields=fields,
            api_key=api_key,
        )
        end_time = time.time()
        print(f"Search time: {end_time - start_time} seconds")
        print(len(results["data"]))
        
        if not results or "data" not in results:
            return []
        
        new_results = []
        for result in results["data"]:
            if result["title"] in self.ban_list:
                continue
            new_results.append(result)
        results = new_results
        
        # 过滤非开放获取的论文
        if need_download:
            paper_candidates = []
            for result in results:
                if (
                    os.path.exists(
                        os.path.join(
                            self.save_dir,
                            f"{result['title']}.pdf"
                        )
                    )
                    and result["title"] not in readed_papers
                ):
                    paper_candidates.append(result)
                elif not result["isOpenAccess"] or not result["openAccessPdf"]:
                    continue
                else:
                    paper_candidates.append(result)
        else:
            paper_candidates = results
        
        # 重排序论文相关性顺序，使用语义相似度
        start_time = time.time()
        if llm and rerank_query:
            rerank_query_embedding = llm.get_embedding(rerank_query)
            rerank_query_embedding = np.array(rerank_query_embedding)
            
            paper_candidates = self.rerank_papers(
                rerank_query_embedding,
                paper_candidates,
                llm,
            )
        else:
            logging.error(f"没有设置论文排序，因此按照默认顺序。")
        end_time = time.time()
        print(f"Rerank time: {end_time - start_time} seconds")
        print(len(paper_candidates))
        paper_candidates.reverse()

        # 根据候选论文下载PDF文件（原串行版本，已注释）
        # start_time = time.time()
        # final_results = []
        
        # while len(final_results) < max_results and len(paper_candidates) > 0:
        #     result = paper_candidates.pop()
        #     article = None
        #     if need_download:
        #         if os.path.exists(
        #             os.path.join(
        #                 self.save_dir,
        #                 f"{result['title']}.pdf"
        #             )
        #         ):
        #             article = await self.read_arxiv_from_path(
        #                 os.path.join(self.save_dir, f"{result['title']}.pdf")
        #             )
        #         else:
        #             pdf_link = result["openAccessPdf"]["url"]
        #             article = await self.read_arxiv_from_link_async(
        #                 pdf_link,
        #                 f"{result['title']}.pdf"
        #             )
        #         if not article:
        #             continue
            
        #     title, abstract, citationCount, year = (
        #         result["title"],
        #         result["abstract"],
        #         result["citationCount"],
        #         result["year"],
        #     )
        #     final_results.append(Result(
        #         title=title,
        #         abstract=abstract,
        #         article=article,
        #         citations_count=citationCount,
        #         year=year
        #     ))
        
        # end_time = time.time()
        # print(f"Download time: {end_time - start_time} seconds")
        # print(len(final_results))
        # return final_results
            
        # for result in paper_candidates:
        #     article = None
        #     if need_download:
        #         if os.path.exists(
        #             os.path.join(
        #                 self.save_dir,
        #                 f"{result['title']}.pdf"
        #             )
        #         ):
        #             article = await self.read_arxiv_from_path(
        #                 os.path.join(
        #                     self.save_dir,
        #                     f"{result['title']}.pdf"
        #                 )
        #             )
        #         else:
        #             pdf_link = result["openAccessPdf"]["url"]
        #             article = await self.read_arxiv_from_link_async(
        #                 pdf_link,
        #                 f"{result['title']}.pdf"
        #             )
        #         if not article:
        #             continue
            
        #     title, abstract, citationCount, year = (
        #         result["title"],
        #         result["abstract"],
        #         result["citationCount"],
        #         result["year"],
        #     )
        #     final_results.append(Result(
        #         title=title,
        #         abstract=abstract,
        #         article=article,
        #         citations_count=citationCount,
        #         year=year
        #     ))
        #     if len(final_results) >= max_results:
        #         break
            
        # end_time = time.time()
        # print(f"Download time: {end_time - start_time} seconds")
        # print(len(final_results))
        # return final_results
        
        # 并发下载PDF文件
        start_time = time.time()
        final_results = []

        if need_download:
            semaphore = asyncio.Semaphore(20)  # 控制并发数

            async def download_item(result):
                async with semaphore:
                    pdf_path = os.path.join(self.save_dir, f"{result['title']}.pdf")
                    if os.path.exists(pdf_path):
                        article = await self.read_arxiv_from_path(pdf_path)
                        return result, article
                    elif result.get("isOpenAccess") and result.get("openAccessPdf"):
                        pdf_link = result["openAccessPdf"]["url"]
                        article = await self.read_arxiv_from_link_async(pdf_link, f"{result['title']}.pdf")
                        return result, article
                    return result, None

            # 持续尝试直到满足数量或耗尽候选
            while len(final_results) < max_results and paper_candidates:
                # 当前批次：取 min(剩余需要数 + 2, 剩余候选) 用于并发
                remaining_needed = max_results - len(final_results)
                batch_size = min(remaining_needed + 2, len(paper_candidates))
                batch = [paper_candidates.pop() for _ in range(batch_size)]
                
                tasks = [download_item(r) for r in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for item in results:
                    if isinstance(item, Exception) or item[1] is None:
                        continue
                    result, article = item
                    final_results.append(Result(
                        title=result["title"],
                        abstract=result["abstract"],
                        article=article,
                        citations_count=result["citationCount"],
                        year=result["year"],
                    ))
                    # ✅ 一旦满足就提前退出内层
                    if len(final_results) >= max_results:
                        break

                # 外层 while 条件控制继续或退出
        else:
            # 不需要下载：直接取前 N 个
            while len(final_results) < max_results and paper_candidates:
                result = paper_candidates.pop()
                final_results.append(Result(
                    title=result["title"],
                    abstract=result["abstract"],
                    article=None,
                    citations_count=result["citationCount"],
                    year=result["year"],
                ))
        
        end_time = time.time()
        print(f"Download time: {end_time - start_time} seconds")
        print(len(final_results))
        return final_results
    
    # 异步搜索相关论文（引用和参考文献）
    async def search_related_paper_async(
        self,
        title: str,
        need_citation: bool = True,
        need_reference: bool = True,
        rerank_query: str | None = None,
        llm: Any | None = None,
        paper_list: List[Result] = [],
        logger: Logger | None = None,
        api_key: str | None = None,
    ) -> List[Result] | None:
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
            logger: 日志记录器
            api_key: Semantic Scholar API密钥
        返回：相关论文的Result对象，如果失败返回None
        """
        if logger:
            logger.info(
                f"Searching for related papers of paper <{title}>; Citation:{need_citation}; Reference:{need_reference}"
            )
        
        fields = [
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
        results = await self.search_papers_async(
            title, limit=3, fields=fields, logger=logger, api_key=api_key,
        )
        
        related_papers = []
        related_papers_title = []
        if not results or "data" not in results:
            if logger:
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
                            os.path.join(self.save_dir, f"{citation['title']}.pdf")
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
                        or citation["title"] in self.ban_list
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
                            os.path.join(self.save_dir, f"{reference['title']}.pdf")
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
                        or reference["title"] in self.ban_list
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
            rerank_query_embedding = llm.get_embedding(rerank_query)
            rerank_query_embedding = np.array(rerank_query_embedding)
            related_papers = self.rerank_papers(rerank_query_embedding, related_papers, llm)
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
        if logger:
            logger.info(f"Found {len(related_papers)} related papers")
        for paper in related_papers:
            url = paper[2]
            article = await self.read_arxiv_from_link_async(url, f"{paper[0]}.pdf")
            if not article:
                continue
            result = Result(
                title=paper[0],
                abstract=paper[1],
                article=article,
                citations_count=paper[3],
                year=paper[4]
            )
            if logger:
                logger.info(f"Successfully found related papers of paper <{title}>")
            return result
        if logger:
            logger.warning(
                f"Failed to find related papers of paper <{title}>; Citation:{need_citation}; Reference:{need_reference}"
            )
        return None
    
    # 异步从PDF链接读取论文内容
    async def read_arxiv_from_link_async(self, pdf_link: str, filename: str) -> dict | None:
        """
        异步从PDF链接读取论文内容
        功能：检查本地是否已有PDF文件，如果没有则下载并解析
        参数：
            pdf_link: PDF文件的下载链接
            filename: 保存的文件名
        返回：解析后的论文内容字典，失败时返回None
        """
        file_path = os.path.join(self.save_dir, filename)  # 构建本地文件路径
        # 如果文件已存在，直接从本地读取
        if os.path.exists(file_path):
            article_dict = await self.read_arxiv_from_path(file_path)
            return article_dict

        # 如果文件不存在，先下载PDF
        result = await self.download_pdf_async(pdf_link, file_path)
        if not result:  # 如果下载失败
            logger.error(f"Failed to download the PDF file: {filename}")
            return None
        try:  # 尝试解析下载的PDF文件
            article_dict = self.read_arxiv_from_path(file_path)
            # article_dict = await self.read_arxiv_from_path(file_path)
            return article_dict
        except Exception as e:  # 如果解析失败
            logger.error(
                f"Failed to read the article from the PDF file: {e}, {filename}"
            )
            return None
    
    # 异步下载PDF文件
    async def download_pdf_async(self, pdf_link: str, save_path: str, user_agent: str = "requests/2.0.0") -> bool:
        """
        异步下载PDF文件
        功能：从网络下载PDF文件并保存到本地
        参数：
            pdf_link: PDF文件的下载链接
            save_path: 本地保存路径
            user_agent: 用户代理字符串，默认"requests/2.0.0"
        返回：下载成功返回True，失败返回False
        """
        if os.path.exists(save_path):
            logger.info(f"The PDF file <{save_path}> already exists.")
            return True
        try:
            dirpath = os.path.dirname(save_path)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            timeout = aiohttp.ClientTimeout(total=120)
            headers = {
                "user-agent": user_agent,
            }
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    pdf_link, headers=headers, allow_redirects=True, ssl=False
                ) as response:
                    response.raise_for_status()
                    content_type = (response.headers.get("content-type") or "").lower()
                    if content_type.startswith("application/pdf"):
                        target_path = save_path
                    else:
                        logger.error(
                                f"Unsupported Content-Type: {content_type}"
                        )
                        return False
                    
                    with open(target_path, "wb") as f:
                        async for chunk in response.content.iter_chunked(81920):
                            f.write(chunk)
            
            logger.info(f"Successfully saved file to: {target_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download the PDF file: {e}, {save_path}")
            return False

