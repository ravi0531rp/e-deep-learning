import asyncio
from pathlib import Path
from typing import Dict, List, Optional
import aiohttp
from loguru import logger

url = "http://127.0.0.1:8001/predict"

async def run_main(
    session: aiohttp.ClientSession, img_path: str) -> Dict:

    form = aiohttp.FormData()
    form.add_field("file", open(img_path, "rb"))

    content = {}
    try:
        async with session.post(url, data=form) as response:
            if response.status == 200:
                content = await response.json()
            else:
                status_code = 1101
                logger.error(f"{status_code}: server returned status {response.status}")

    except aiohttp.client_exceptions.ClientConnectorError as e:
        status_code = 1101
        logger.error(f"{status_code}: Unable to access url")

    except Exception as e:
        status_code = 1002
        logger.error(f"{status_code}: {repr(e)}")

    return content



async def main(img_path):
    tasks = []
    async with aiohttp.ClientSession() as session:
        tasks.append(run_main(session, img_path))
        results = await asyncio.gather(*tasks)

    return results

if __name__ == "__main__":
    res = asyncio.run(main("/home/heisenberg/Downloads/horse.jpg"))
    logger.debug(res)