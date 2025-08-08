# membership-inference

## Setup

Create conda environment:
```bash
conda create -n mia python=3.10
conda activate mia
pip install -r requirements.txt
```

Create a `.env` file with the following variables:

```
OPENAI_API_KEY=your_openai_api_key_here
CACHE_PATH=/path/to/your/cache
```

Then get the dataset splits by running:

```
python3 datasets.preprocess.py
```