from code.user_secrets import OPENAI_API_KEY
from openai import OpenAI
import time
import logging
from code.helper.generation.api_request_parallel_processor import process_api_requests
from IPython import embed

requests_limits_dict = {
    "gpt-3.5-turbo-0125": {
        "max_requests_per_minute": 10000,
        "max_tokens_per_minute": 30000000
    },
    "gpt-4-0613": {
        "max_requests_per_minute": 10000 // 2,
        "max_tokens_per_minute": 1000000 // 2
    },
    "gpt-3.5-turbo-instruct": {
        "max_requests_per_minute": (3500 * 4) // 5,
        "max_tokens_per_minute": (90000 * 4) // 5
    }
}
requests_limits_dict["gpt-3.5-turbo-1106"] = requests_limits_dict["gpt-3.5-turbo-0125"]
requests_limits_dict["gpt-4-0314"] = requests_limits_dict["gpt-4-0613"]

# Store the type of request we need to make for different models
requests_url_dict = {
    "gpt-3.5-turbo-0125": "https://api.openai.com/v1/chat/completions",
    "gpt-3.5-turbo-instruct": "https://api.openai.com/v1/completions"
}
for model in ["gpt-4-0613", "gpt-3.5-turbo-1106", "gpt-4-0314"]:
    requests_url_dict[model] = requests_url_dict["gpt-3.5-turbo-0125"]

async def openai_parallel_generate(requests, 
                                   args, 
                                   max_requests_per_minute=10000, 
                                   max_tokens_per_minute=30000000,
                                   request_url="https://api.openai.com/v1/chat/completions"):
    start_time = time.time()
    results = await process_api_requests(
        requests=requests,
        request_url=request_url,
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


