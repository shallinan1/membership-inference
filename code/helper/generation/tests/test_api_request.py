from code.helper.generation.api_request_parallel_processor import process_api_requests
import asyncio
import logging
from code.user_secrets import OPENAI_API_KEY
import time
from IPython import embed

async def main():
    # Define your chat completion requests
    requests = [
        {
            "model": "gpt-4o-2024-11-20",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Generate a PhD level thesis on gender biases throughout modern cultures."}
            ],
            "max_tokens": 2048,
            "temperature": 0.1,
            "metadata": {"request_id": 1}  # Optional metadata
        },
        {
            "model": "gpt-4o-2024-11-20",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain quantum computing as verbosely as possible. List many theories and potential explanations."}
            ],
            "max_tokens": 2048,
            "temperature": 0.1,
            "metadata": {"request_id": 2}  # Optional metadata
        },
        {
            "model": "gpt-4o-2024-11-20",
            "messages": [{"role": "user", "content": "Generate a long story with 20 chapters, 500 words per chapter"}],
            "max_tokens": 2048,
            "temperature": 0.1,
            "metadata": {"id": 1},
        },
    ] * 20

    start_time = time.time()

    # Call the process_api_requests function
    results = await process_api_requests(
        requests=requests,
        request_url="https://api.openai.com/v1/chat/completions",  # gpt-4o-2024-11-20 chat completions endpoint
        api_key=OPENAI_API_KEY,  # Replace with your OpenAI API key
        max_requests_per_minute=10000,  # Adjust based on your rate limits
        max_tokens_per_minute=30000000,  # Adjust based on your rate limits
        token_encoding_name="cl100k_base",  # Token encoding for gpt-4o-2024-11-20
        max_attempts=5,  # Number of retries for failed requests
        logging_level=logging.WARNING  # Set logging level
    )

    # Stop timing
    end_time = time.time()

    # Print results and timing information
    print(f"Total generation time: {end_time - start_time:.2f} seconds for {len(requests)} generations")

    return results

if __name__ == "__main__":
    results = asyncio.run(main())
    embed()

"""
python3 -m code.helper.generation.tests.test_api_request
"""

