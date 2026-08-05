# Neural Sentinel

Multi-agent financial fraud detection system for Nepali banking channels.



### Primary — uv (recommended for local development)

```bash
# Install uv
pip install uv

# Create venv and install dependencies
uv venv
uv pip install -r requirements.txt

# Run a script
uv run python <script>
```

### Secondary — pip

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/macOS
pip install -r requirements.txt
```

### Secondary — conda

```bash
conda env create -f environment.yml
conda activate neural-sentinel
```

### Kaggle

Use `!pip install -q <package>` at the top of each notebook for any packages not pre-installed on Kaggle. Do not use a local venv on Kaggle.
