from code.user_secrets import OPENAI_API_KEY
from openai import OpenAI
import time
import logging
from code.helper.generation.api_request_parallel_processor import process_api_requests
from IPython import embed

async def openai_generations(requests, args, max_requests_per_minute=10000, max_tokens_per_minute=30000000):
    start_time = time.time()
    
    results = await process_api_requests(
        requests=requests,
        request_url="https://api.openai.com/v1/chat/completions",
        api_key=OPENAI_API_KEY, 
        max_requests_per_minute=max_requests_per_minute, 
        max_tokens_per_minute=max_tokens_per_minute,  
        token_encoding_name="cl100k_base",
        max_attempts=5, 
        logging_level=logging.WARNING 
    )

    end_time = time.time()
    print(f"Total generation time: {end_time - start_time:.2f} seconds for {len(requests)} samples")

    return results

if __name__ == "__main__":
    # TODO write a couple tests to ensure this works correctly
    embed()


