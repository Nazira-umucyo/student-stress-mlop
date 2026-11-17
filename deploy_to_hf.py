"""Upload the project to a Hugging Face Space (Docker), which builds and hosts it."""
from pathlib import Path

from huggingface_hub import HfApi, login

HF_USERNAME = "Umucyo"  
SPACE_ID = f"{HF_USERNAME}/student-stress-mlop"

login()  # asks for your Hugging Face access token the first time
api = HfApi()
api.create_repo(SPACE_ID, repo_type="space", space_sdk="docker", exist_ok=True)

api.upload_folder(
    folder_path=".",
    repo_id=SPACE_ID,
    repo_type="space",
    commit_message="Deploy app",
    ignore_patterns=[".git/*", "**/__pycache__/*", "*.pyc", "notebook/*",
                     "data/stress.db", "models/archive/*", ".env", "README.md"],
)

# Hugging Face reads the Space settings from the top of README.md
header = """---
title: Student Stress Predictor
emoji: 🧠
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

"""
readme = Path("README.md").read_text(encoding="utf-8")
api.upload_file(path_or_fileobj=(header + readme).encode("utf-8"),
                path_in_repo="README.md", repo_id=SPACE_ID, repo_type="space",
                commit_message="Add Space README")

print(f"\nUploaded! Build progress: https://huggingface.co/spaces/{SPACE_ID}")
print(f"Live app (once built): https://{SPACE_ID.replace('/', '-').lower()}.hf.space")