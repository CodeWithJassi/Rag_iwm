---
Step 1 — Install system packages

sudo apt update && sudo apt install -y python3.12-venv postgresql postgresql-16-pgvector

Step 2 — Install Ollama in WSL (for embeddings, ~270 MB)

curl -fsSL https://ollama.com/install.sh | sudo sh

Step 3 — Create venv and install Python dependencies

cd /mnt/e/for_ubuntu/claude_dev/rag_project_iwm
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Step 4 — Start PostgreSQL and create the database

sudo pg_ctlcluster 16 main start
sudo -u postgres psql -c "CREATE USER iwm WITH PASSWORD 'iwm' CREATEDB;"
sudo -u postgres psql -c "CREATE DATABASE iwm_rag OWNER iwm;"

Step 5 — Start Ollama and pull the embedding model

ollama serve &
sleep 2
ollama pull nomic-embed-text

Step 6 — Start the app

source venv/bin/activate
cd backend && uvicorn main:app --reload

Then open http://localhost:8000 in your browser.

---

#########################################################################

# switch embedding model
How to switch models

# 1. Set the model in .env
echo 'EMBED_MODEL=mxbai-embed-large:latest' >> .env

# 2. Restart — the column auto-migrates, existing vectors are dropped
cd backend && uvicorn main:app --reload

# 3. Re-ingest every report (the old vectors are gone)

The startup logs will show a big warning box confirming the migration happened.

#########################################################################

To use the new chunking strategy, set CHUNKING_STRATEGY=structure in .env.