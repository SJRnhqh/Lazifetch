# src/lazifetch/model/SemanticSearcher.py
# 导入外部库
from typing import List, Any
from logging import Logger
from numpy import ndarray
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
from .Result import Result


logger = logging.getLogger(__name__)


class SemanticSearcher:
    def __init__(self, save_dir: str = "papers/", ban_list: List[str] = []) -> None:
        self.save_dir = save_dir
        self.ban_list = ban_list

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
                logger.warning(
                    f"Request failed with status code {response.status_code}: begin to retry"
                )

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
                    api_key,
                )
            else:
                logger.error(
                    f"Request failed with status code {response.status_code}: {response.text}"
                )
                return None
        except requests.RequestException as e:
            logger.error(f"An error occurred: {e}")
            return None

    def cal_cosine_similarity(self, vec1: ndarray, vec2: ndarray) -> float:
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def _cal_cosine_similarity_matric(
        self, matric1: ndarray, matric2: ndarray
    ) -> List[float]:
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

    def rerank_papers(
        self, query_embedding: ndarray, paper_list: List[dict], llm
    ) -> List[dict]:
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

        paper_list = sorted(
            zip(paper_list, scores), key=lambda x: x[1], reverse=True
        )

        paper_list = [paper[0] for paper in paper_list]
        return paper_list

    async def read_arxiv_from_path(self, pdf_path: str) -> Result | None:
        if not os.path.exists(pdf_path):
            logger.error(f"The PDF file <{pdf_path}> does not exist.")
            return None
        try:
            article_dict = scipdf.parse_pdf_to_dict(pdf_path)
            logger.info(f"Successfully parsed the PDF file: {article_dict}")
            return article_dict
        except Exception as e:
            logger.error(
                f"Failed to read the article from the PDF file: {e}, {pdf_path}"
            )
            return None

    def read_paper_title_abstract(self, article: dict) -> tuple[str, str]:
        title = article["title"]
        abstract = article["abstract"]
        paper_content = f"""
            Title: {title}
            Abstract: {abstract}
        """
        return paper_content

    def read_paper_title_abstract_introduction(self, article: dict) -> tuple[str, str]:
        title = article["title"]
        abstract = article["abstract"]
        introduction = article["sections"][0]["text"]
        paper_content = f"""
            Title: {title}
            Abstract: {abstract}
            Introduction: {introduction}
        """
        return paper_content

    def read_paper_content(self, article: dict) -> str:
        paper_content = self.read_paper_title_abstract(article)
        for section in article["sections"]:
            paper_content += f"section: {section['heading']}\n content: {section['text']}\n ref_ids: {section['publication_ref']}\n"
        return paper_content

    def read_paper_content_with_ref(self, article: dict) -> str:
        paper_content = self.read_paper_content(article)
        paper_content += "<References>\n"
        for refer in article["references"]:
            ref_id = refer["ref_id"]
            title = refer["title"]
            year = refer["year"]
            paper_content += f"Ref_id:{ref_id} Title: {title} Year: ({year})\n"
        paper_content += "</References>\n"
        return paper_content

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

        if need_download:
            paper_candidates = []
            for result in results:
                if (
                    os.path.exists(
                        os.path.join(self.save_dir, f"{result['title']}.pdf")
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

        start_time = time.time()
        final_results = []

        if need_download:
            semaphore = asyncio.Semaphore(10)

            async def download_item(result):
                async with semaphore:
                    pdf_path = os.path.join(self.save_dir, f"{result['title']}.pdf")
                    if os.path.exists(pdf_path):
                        article = await self.read_arxiv_from_path(pdf_path)
                        return result, article
                    elif result.get("isOpenAccess") and result.get("openAccessPdf"):
                        pdf_link = result["openAccessPdf"]["url"]
                        article = await self.read_arxiv_from_link_async(
                            pdf_link, f"{result['title']}.pdf"
                        )
                        return result, article
                    return result, None

            while len(final_results) < max_results and paper_candidates:
                remaining_needed = max_results - len(final_results)
                batch_size = min(remaining_needed, len(paper_candidates))
                batch = [paper_candidates.pop() for _ in range(batch_size)]

                tasks = [download_item(r) for r in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for item in results:
                    if isinstance(item, Exception) or item[1] is None:
                        continue
                    result, article = item
                    final_results.append(
                        Result(
                            title=result["title"],
                            abstract=result["abstract"],
                            article=article,
                            citations_count=result["citationCount"],
                            year=result["year"],
                        )
                    )
                    if len(final_results) >= max_results:
                        break

        else:
            while len(final_results) < max_results and paper_candidates:
                result = paper_candidates.pop()
                final_results.append(
                    Result(
                        title=result["title"],
                        abstract=result["abstract"],
                        article=None,
                        citations_count=result["citationCount"],
                        year=result["year"],
                    )
                )

        end_time = time.time()
        print(f"Download time: {end_time - start_time} seconds")
        print(len(final_results))
        return final_results

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
            title,
            limit=3,
            fields=fields,
            logger=logger,
            api_key=api_key,
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
            related_papers = self.rerank_papers(
                rerank_query_embedding, related_papers, llm
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
                year=paper[4],
            )
            if logger:
                logger.info(f"Successfully found related papers of paper <{title}>")
            return result
        if logger:
            logger.warning(
                f"Failed to find related papers of paper <{title}>; Citation:{need_citation}; Reference:{need_reference}"
            )
        return None

    async def read_arxiv_from_link_async(
        self, pdf_link: str, filename: str
    ) -> dict | None:
        file_path = os.path.join(self.save_dir, filename)
        if os.path.exists(file_path):
            article_dict = await self.read_arxiv_from_path(file_path)
            return article_dict

        result = await self.download_pdf_async(pdf_link, file_path)
        if not result:
            logger.error(f"Failed to download the PDF file: {filename}")
            return None
        try:
            article_dict = self.read_arxiv_from_path(file_path)
            return article_dict
        except Exception as e:
            logger.error(
                f"Failed to read the article from the PDF file: {e}, {filename}"
            )
            return None

    async def download_pdf_async(
        self, pdf_link: str, save_path: str, user_agent: str = "requests/2.0.0"
    ) -> bool:
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
                        logger.error(f"Unsupported Content-Type: {content_type}")
                        return False

                    with open(target_path, "wb") as f:
                        async for chunk in response.content.iter_chunked(81920):
                            f.write(chunk)

            logger.info(f"Successfully saved file to: {target_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download the PDF file: {e}, {save_path}")
            return False
