# tds_project1

# LLM Code Deployment Student API

## Overview

This project implements a **student-side API** for Project 1: LLM Code Deployment.  

The API receives task requests from instructors, generates minimal applications using an LLM, pushes the code to GitHub, deploys via GitHub Pages, and reports the repository and deployment URLs back to the evaluation system.

---

## Features

- **FastAPI Endpoint**: Accepts POST requests with JSON payloads containing task briefs, attachments, and metadata.
- **Secret Validation**: Verifies the student-provided secret for security.
- **LLM Code Generation**: Uses an LLM (e.g., OpenAI GPT) to generate code based on the task brief and attachments.
- **GitHub Integration**: 
  - Creates a unique repository for each task.
  - Pushes generated files.
  - Adds an MIT LICENSE and README.md.
  - Enables GitHub Pages for deployment.
- **Round 2 Support**: Handles subsequent task modifications and redeployment based on instructor feedback.
- **Evaluation Reporting**: Sends repository and Pages URLs back to the instructor evaluation API.

---

## Folder Structure
```
student/
│
├─ main.py # FastAPI app, entrypoint
├─ .env # Environment variables: GITHUB_TOKEN, SECRET, OPENAI_API_KEY
└─ requirements.txt
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/<your-username>/llm-code-deployment.git
cd llm-code-deployment/student
```
2. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
.venv\Scripts\activate    # Windows

pip install -r requirements.txt
```
3. Create a .env file with your secrets:
```ini
GITHUB_TOKEN=ghp_xxxxxxxxxxxxx
SECRET=your_secret
OPENAI_API_KEY=sk-xxxxxxxxxxxx
```


## Usage

Start the FastAPI server:
```bash
uvicorn main:app --reload
```

Example request

Send a POST request to /handle_task with a JSON payload:
```bash
curl -X POST http://127.0.0.1:8000/handle_task \
-H "Content-Type: application/json" \
-d @examples/round1_request.json
```

The API will:

1. Validate the secret.
2. Generate code using the LLM.
3. Create a GitHub repo and push files.
4. Enable GitHub Pages.
Respond with:
```json
{
  "repo_url": "https://github.com/<your-username>/<repo-name>",
  "pages_url": "https://<your-username>.github.io/<repo-name>/"
}
```


## Round 2 Support

For subsequent tasks:
1. Accept a second POST request ({"round": 2}) from the instructor.
2. Verify the secret.
3. Update the repository and redeploy GitHub Pages based on the new brief.
4. Update README.md and other necessary files.
5. Report back to the evaluation URL with updated repo details.


## Testing

Run the included pytest tests:
```bash
pytest tests/
```

This validates:

1. GitHub integration (github.py)
2. LLM code generation (llm.py)
3. File utilities (files.py)


## Notes & Best Practices

1. Ensure GITHUB_TOKEN has repo and workflow permissions.
2. Avoid committing secrets to the repository (use .env and gitignore).
3. Generated apps must be minimal but functional according to the task brief.
4. Round 2 requests may require code refactoring, feature additions, or bug fixes.


## License

This project is licensed under the MIT License. See LICENSE
 for details.

```yaml

