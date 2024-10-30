# membership-inference

Please make a file named `user_secrets.py` with the following variables

```
OPENAI_API_KEY= # Your OpenAI API key
CACHE_PATH= # Your cache path
```

Then get the dataset splits by running:

```
python3 datasets.preprocess.py
```