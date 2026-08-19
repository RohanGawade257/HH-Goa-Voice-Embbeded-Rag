That's just an environment dependency issue. Your code isn't failing yet. The active .venv doesn't have sentence-transformers installed, even though your project was migrated to it.

Run this inside the same activated .venv:

python -m pip install "sentence-transformers>=2.7.0"

Then verify:

python -c "from sentence_transformers import SentenceTransformer; print('SentenceTransformers OK')"

You should get:

SentenceTransformers OK

Then rerun:

python benchmarks\benchmark_qwen_api_manual.py --allow-live-api
If it still fails

Check that python and pip are pointing to the same .venv:

python -c "import sys; print(sys.executable)"
python -m pip --version

Both should point somewhere under:

HH-Goa-Rag\.venv\
One important thing

Your earlier agent report claimed:

sentence-transformers>=2.7.0 was added to requirements.txt

but your current environment clearly doesn't have it. That means the dependency migration was committed to the project files but wasn't installed into this virtual environment.

After installation, don't change anything else yet. Run the Qwen benchmark and give me the output. That's the measurement we actually need now.

Manual live API benchmark file: benchmarks/benchmark_qwen_api_manual.py; requires --allow-live-api.