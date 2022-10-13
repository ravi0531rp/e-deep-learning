## python test-api-async.py -u http://127.0.0.1:8001/predict -f ./test.csv -c 4 -t 4 -i 1 -l ./logs/log.log
import time
import asyncio
import aiohttp
import pandas as pd
from aiohttp import ClientResponseError
from loguru import logger
import argparse


my_parser = argparse.ArgumentParser()
my_parser.add_argument("-u", "--url", type=str, required=True)  # url for the endpoint
my_parser.add_argument(
    "-f", "--file", type=str, required=True
)  
my_parser.add_argument(
    "-c", "--concurrency", type=int, default=4
)  
my_parser.add_argument(
    "-t", "--total", type=int, default=4
)  
my_parser.add_argument(
    "-i", "--iterations", type=int, default=1
)  
my_parser.add_argument(
    "-l", "--log", type=str, default="./logs/log.log"
)  

args = vars(my_parser.parse_args())


url = args["url"]
csv_path = args["file"]
logger_file_name = args["log"]
n_iterations = args["iterations"]
n_concurrency = args["concurrency"]
n_requests = args["total"]
logger.add(logger_file_name, level="DEBUG", rotation="100 MB")


my_timeout = aiohttp.ClientTimeout(
    total=None, 
    sock_connect=200000, 
    sock_read=200000, 
)

client_args = dict(trust_env=True, timeout=my_timeout)
semaphore = asyncio.Semaphore(2)  


async def gather_with_concurrency(n, *tasks):
    global semaphore
    semaphore = asyncio.Semaphore(n)  

    async def sem_task(task):
        async with semaphore:
            return await task

    return await asyncio.gather(*(sem_task(task) for task in tasks))


async def fetch(session, url, img_path):
    """This function fetches the response from the Endpoint
    Args:
        session (Session): session Object
        url (str): Url to hit
        payload (dict): The lat-lon pair
    Returns:
        resp (dict): The received Response
    """

    form = aiohttp.FormData()
    form.add_field("file", open(img_path, "rb"))

    resp = None
    try:
        async with session.post(url, data=form) as response:
            resp = await response.json()
    except ClientResponseError as e:
        logger.warning(f" Error {e.status}")
    except asyncio.TimeoutError:
        logger.warning("Timeout")
    except Exception as e:
        logger.warning(f" Error {e}")
    else:
        return resp

    return resp


async def fetch_async(url, images, n_tasks=10):
    tasks = []
    connector = aiohttp.TCPConnector(force_close=True)
    async with aiohttp.ClientSession(connector=connector, **client_args) as session:
        for j in range(len(images)):
            task = asyncio.ensure_future(fetch(session, url, images[j]))
            tasks.append(task)
        responses = await gather_with_concurrency(n_tasks, *tasks)
    logger.info(responses)
    return responses


if __name__ == "__main__":
    try:
        df = pd.read_csv(csv_path)
        images = df["image"]
        logger.info({"url":url, "n_requests":min(len(images),n_requests) , "n_concurrency":n_concurrency, 'cpu': 8, 'memory': 8 })

    except Exception as e:
        logger.error(f"Error : {e}")

    length = len(df)

    time_consumed_list = []
    for n_iter in range(n_iterations):
        logger.info(f"Started Iteration {n_iter + 1}")
        
        start_time = time.perf_counter()
        loop = asyncio.get_event_loop()
        future = asyncio.ensure_future(
            fetch_async(url, images, n_concurrency)
        )
        loop.run_until_complete(future)

        time_consumed = time.perf_counter() - start_time
        logger.info(f"Total time for request is {str(time_consumed)}")
        time_consumed_list.append(time_consumed)

        time.sleep(5)

    logger.success(
        f"For {n_iterations} iterations with Total Events : {n_requests} , Concurrent Events : {n_concurrency} , Average time is {sum(time_consumed_list)/len(time_consumed_list)} ; Max time is {max(time_consumed_list)} ; Min time is {min(time_consumed_list)}"
    )