# main.py
# 导入外部库
from dotenv import load_dotenv
import logging
import os
import asyncio
import time


# 导入内部库
from model.SemanticSearcher import SemanticSearcher


def setup_logging():
    """设置日志记录"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    setup_logging()
    load_dotenv()
    
    api_key = os.environ.get("SEMANTIC_SEARCH_API_KEY")
    
    if not api_key:
        raise ValueError("SEMANTIC_SEARCH_API_KEY is not set")
    
    logger = logging.getLogger(__name__)
    
    searcher = SemanticSearcher(
        save_dir="papers/",
        ban_list=[]
    )


    query = "Machine Learning"
    
    logger.info(f"开始搜索论文: {query}")
    
    # 记录开始时间
    start_time = time.time()
    logger.info("记录开始时间")
    
    async def test_search():
        try:
            # 搜索并下载论文
            logger.info("调用 search_async 方法...")
            results =  await searcher.search_async(
                query=query,
                max_results=5,
                need_download=True,
                api_key=api_key
            )
            
            # if results:
            #     logger.info(f"成功获取 {len(results)} 篇论文")
            #     for i, paper in enumerate(results):
            #         logger.info(f"  论文 {i+1}: {paper.title[:50]}{'...' if len(paper.title) > 50 else ''}")
            # else:
            #     logger.warning("未找到或下载失败")
            # download_duration = time.time() - start_time
            # logger.info(f"搜索和下载总耗时: {download_duration:.2f} 秒")
        except Exception as e:
            logger.error(f"下载过程中出现错误: {e}")
    
    asyncio.run(test_search())

if __name__ == "__main__":
    main()