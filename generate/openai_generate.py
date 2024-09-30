import openai
import time
from user_secrets import OPENAI_API_KEY

# Create client with openAI key
openai.api_key = OPENAI_API_KEY

def get_gpt_output(query, model='davinci-002', temperature=1.0, max_tokens=256, top_p=0.9, n=1):
    attempts = 1
    while attempts <= 20:
        try:
            if model == 'gpt-3.5-turbo-instruct' or any([model.startswith(x) for x in ['babbage', 'davinci']]):
                response = openai.Completion.create(
                    model=model,
                    prompt=query,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    n=n,
                    logprobs=True
                )
                return response
            else:
                messages = [{"role": "user", "content": query}]

                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    n=n,
                    logprobs=True
                )
                return response
        except Exception as e:
            attempts += 1
            print(f"Service unavailable, retrying in 10 seconds ({attempts}/5): {e}")
            time.sleep(10)
    else:
        return None

if __name__ == "__main__":
    # Testing
    from IPython import embed

    output = get_gpt_output("Complete the following sentence: When I",
                            model="gpt-4o-2024-05-13")

    embed()


