# Generation Module

This module contains utilities for generating text using various APIs, including OpenAI and vLLM.

## Configuration

### Rate Limits

Before using the OpenAI generation utilities, you need to configure the rate limits in `rate_limits.py` according to your OpenAI account's limits.

1. Visit https://platform.openai.com/account/rate-limits to check your current rate limits
2. Update the values in `rate_limits.py` for the models you plan to use
3. The `SAFETY_MARGIN` factor (currently 0.75) is applied to prevent hitting actual API limits

**Important**: The default values in `rate_limits.py` are examples and may not match your account's actual limits. Using incorrect rate limits may result in API errors or suboptimal performance.

## Files

- `openai_parallel_generate.py` - Parallel text generation using OpenAI API
- `rate_limits.py` - Rate limit configurations for different OpenAI models
- `api_request_parallel_processor.py` - Core parallel request processing logic
- `generate_utils.py` - Utility functions for text generation
- `vllm_generate.py` - Text generation using vLLM