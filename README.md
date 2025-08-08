# membership-inference

## Setup

Create conda environment:
```bash
conda create -n mia python=3.10
conda activate mia
pip install -r requirements.txt
```

Please make a file named `user_secrets.py` with the following variables

```
OPENAI_API_KEY= # Your OpenAI API key
CACHE_PATH= # Your cache path
```

Then get the dataset splits by running:

```
python3 datasets.preprocess.py
```