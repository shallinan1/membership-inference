import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
import time
import logging
from code.helper.generation.api_request_parallel_processor import process_api_requests
from IPython import embed
from src.generation.rate_limits import requests_limits_dict, requests_url_dict

async def openai_parallel_generate(requests,
                                   max_requests_per_minute=10000, 
                                   max_tokens_per_minute=30000000,
                                   request_url="https://api.openai.com/v1/chat/completions"):
    """
    Generate text using OpenAI API with parallel request processing.
    
    Args:
        requests: List of API request dictionaries to process in parallel
        max_requests_per_minute: Maximum number of requests per minute (rate limiting)
        max_tokens_per_minute: Maximum number of tokens per minute (rate limiting) 
        request_url: OpenAI API endpoint URL (chat/completions or completions)
        
    Returns:
        List of API response results corresponding to input requests
        
    Note:
        Uses rate limiting and retry logic to handle API constraints.
        Default rate limits should be adjusted based on your OpenAI account limits.
    """
    # Track timing for performance monitoring
    start_time = time.time()
    
    # Process requests in parallel with rate limiting and retry logic
    results = await process_api_requests(
        requests=requests,
        request_url=request_url,
        api_key=OPENAI_API_KEY, 
        max_requests_per_minute=max_requests_per_minute, 
        max_tokens_per_minute=max_tokens_per_minute,  
        token_encoding_name="cl100k_base",  # Encoding for token counting
        max_attempts=5,  # Retry failed requests up to 5 times
        logging_level=logging.WARNING  # Only log warnings and errors
    )

    # Report total processing time
    end_time = time.time()
    print(f"Total generation time: {end_time - start_time:.2f} seconds for {len(requests)} samples")

    return results

if __name__ == "__main__":
    import asyncio
    
    async def test_basic_functionality():
        """Simple test to verify the function works correctly."""
        # Test with a couple of simple requests
        sample_requests = [
            {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Say 'Hello World'"}]},
            {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Count to 3"}]}
        ]
        
        print("Testing openai_parallel_generate with 2 sample requests...")
        
        try:
            results = await openai_parallel_generate(
                requests=sample_requests,
                max_requests_per_minute=requests_limits_dict["gpt-3.5-turbo-0125"]["max_requests_per_minute"],
                max_tokens_per_minute=requests_limits_dict["gpt-3.5-turbo-0125"]["max_tokens_per_minute"],
                request_url=requests_url_dict["gpt-3.5-turbo-0125"]
            )
            
            print(f"✓ Successfully processed {len(results)} requests")
            for i, result in enumerate(results):
                if result and 'choices' in result:
                    print(f"  Request {i+1}: {result['choices'][0]['message']['content'][:50]}...")
                else:
                    print(f"  Request {i+1}: Error or empty response")
                    
        except Exception as e:
            print(f"✗ Test failed: {e}")
    
    # Run the test
    asyncio.run(test_basic_functionality())
    
    # Interactive development environment
    embed()


