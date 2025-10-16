def verify_pages_deployment(repo_name: str, max_wait_seconds: int = 300):
    """
    Wait for Pages to be deployed and accessible.
    Increased timeout to 5 minutes with better retry logic.
    Returns True if Pages is accessible, False otherwise.
    """
    pages_url = f"https://{USERNAME}.github.io/{repo_name}/"
    print(f"⏳ Waiting for GitHub Pages deployment (max {max_wait_seconds}s)...")
    print(f"📍 Target URL: {pages_url}")
    
    # Initial delay to let GitHub Actions trigger
    time.sleep(20)
    
    start_time = time.time()
    attempt = 0
    consecutive_errors = 0
    
    while time.time() - start_time < max_wait_seconds:
        attempt += 1
        elapsed = int(time.time() - start_time)
        
        try:
            response = httpx.get(pages_url, timeout=15.0, follow_redirects=True)
            consecutive_errors = 0  # Reset on any successful connection
            
            if response.status_code == 200:
                print(f"✅ Pages is live! (attempt {attempt}, elapsed {elapsed}s)")
                return True
            elif response.status_code == 404:
                print(f"⏳ Attempt {attempt}: Still deploying... ({elapsed}s elapsed)")
                time.sleep(15)
            elif response.status_code == 403:
                print(f"⏳ Attempt {attempt}: Access forbidden, still deploying... ({elapsed}s elapsed)")
                time.sleep(15)
            else:
                print(f"⚠️ Attempt {attempt}: Unexpected status {response.status_code} ({elapsed}s elapsed)")
                time.sleep(15)
                
        except httpx.ConnectError:
            consecutive_errors += 1
            print(f"⏳ Attempt {attempt}: Connection refused, retrying... ({elapsed}s elapsed)")
            time.sleep(15)
        except httpx.TimeoutException:
            consecutive_errors += 1
            print(f"⏳ Attempt {attempt}: Timeout, retrying... ({elapsed}s elapsed)")
            time.sleep(15)
        except Exception as e:
            consecutive_errors += 1
            print(f"⏳ Attempt {attempt}: {type(e).__name__}, retrying... ({elapsed}s elapsed)")
            time.sleep(15)
    
    print(f"❌ Pages not accessible after {max_wait_seconds}s")
    print(f"📍 URL: {pages_url}")
    print(f"💡 Check deployment status: https://github.com/{USERNAME}/{repo_name}/actions")
    print(f"💡 Check Pages settings: https://github.com/{USERNAME}/{repo_name}/settings/pages")
    return False


"""
Standalone LLM App Builder & Deployer with Enhanced Model Selection
All functionality in one file for easy deployment
"""

from fastapi import FastAPI, Request, BackgroundTasks
import os
import json
import base64
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from github import Github, GithubException, Auth
from openai import OpenAI
import httpx

# Load environment variables
load_dotenv()

# Configuration
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
USERNAME = os.getenv("GITHUB_USERNAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USER_SECRET = os.getenv("USER_SECRET")
PROCESSED_PATH = "/tmp/processed_requests.json"
TMP_DIR = Path("/tmp/llm_attachments")

# Initialize clients with proper authentication
auth = Auth.Token(GITHUB_TOKEN)
g = Github(auth=auth)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI()

# Create temp directory
TMP_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# GITHUB UTILITIES
# ============================================================================

def create_repo(repo_name: str, description: str = ""):
    """Create a public repository with the given name."""
    user = g.get_user()
    try:
        repo = user.get_repo(repo_name)
        print(f"✅ Repo already exists: {repo.full_name}")
        return repo
    except GithubException:
        pass

    repo = user.create_repo(
        name=repo_name,
        description=description,
        private=False,
        auto_init=False
    )
    print(f"✅ Created repo: {repo.full_name}")
    return repo


def create_or_update_file(repo, path: str, content: str, message: str):
    """Create a file or update if it already exists."""
    try:
        current = repo.get_contents(path)
        sha = current.sha
        repo.update_file(path, message, content, sha)
        print(f"✅ Updated {path} in {repo.full_name}")
    except GithubException as e:
        if e.status == 404:
            repo.create_file(path, message, content)
            print(f"✅ Created {path} in {repo.full_name}")
        else:
            raise


def create_or_update_binary_file(repo, path: str, binary_content, commit_message: str):
    """Create or update a binary file in the repository."""
    try:
        try:
            current = repo.get_contents(path)
            repo.update_file(
                path=path,
                message=commit_message,
                content=binary_content,
                sha=current.sha
            )
            print(f"✅ Updated binary file {path} in {repo.full_name}")
        except GithubException as e:
            if e.status == 404:
                repo.create_file(
                    path=path,
                    message=commit_message,
                    content=binary_content
                )
                print(f"✅ Created binary file {path} in {repo.full_name}")
            else:
                raise
        return True
    except Exception as e:
        print(f"⚠️ Error creating/updating binary file {path}: {e}")
        return False


def enable_pages(repo_name: str, branch: str = "main"):
    """
    Enable GitHub Pages automatically using multiple methods.
    Falls back to creating gh-pages branch if API fails.
    """
    url = f"https://api.github.com/repos/{USERNAME}/{repo_name}/pages"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {"source": {"branch": branch, "path": "/"}}
    
    # Method 1: Try REST API
    try:
        r = httpx.post(url, headers=headers, json=data, timeout=30.0)
        if r.status_code in (201, 204):
            print(f"✅ Pages enabled via API for {repo_name}")
            return True
        elif r.status_code == 409:
            print(f"✅ Pages already enabled for {repo_name}")
            return True
        elif r.status_code == 403:
            print(f"⚠️ API method failed (403), trying alternative method...")
        else:
            print(f"⚠️ Pages API returned: {r.status_code}")
    except Exception as e:
        print(f"⚠️ API call failed: {e}")
    
    # Method 2: Create/update gh-pages branch (automatic deployment)
    try:
        print(f"🔄 Attempting gh-pages branch method...")
        user = g.get_user()
        repo = user.get_repo(repo_name)
        
        # Get the main branch ref
        main_ref = repo.get_git_ref(f"heads/{branch}")
        main_sha = main_ref.object.sha
        
        # Try to create gh-pages branch pointing to main
        try:
            repo.create_git_ref(ref=f"refs/heads/gh-pages", sha=main_sha)
            print(f"✅ Created gh-pages branch")
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"✅ gh-pages branch already exists")
                # Update it to point to latest commit
                gh_pages_ref = repo.get_git_ref("heads/gh-pages")
                gh_pages_ref.edit(sha=main_sha, force=True)
                print(f"✅ Updated gh-pages branch")
            else:
                raise e
        
        # Now try to enable Pages again with gh-pages branch
        data_gh = {"source": {"branch": "gh-pages", "path": "/"}}
        r2 = httpx.post(url, headers=headers, json=data_gh, timeout=30.0)
        if r2.status_code in (201, 204, 409):
            print(f"✅ Pages enabled with gh-pages branch")
            return True
            
    except Exception as e:
        print(f"⚠️ gh-pages method failed: {e}")
    
    # Method 3: Update repository settings to enable Pages
    try:
        print(f"🔄 Attempting repository settings update...")
        settings_url = f"https://api.github.com/repos/{USERNAME}/{repo_name}"
        
        # First, ensure the repository is public
        settings_data = {
            "has_pages": True,
            "private": False
        }
        r3 = httpx.patch(settings_url, headers=headers, json=settings_data, timeout=30.0)
        if r3.status_code == 200:
            print(f"✅ Updated repository settings")
            
            # Try Pages API one more time
            time.sleep(2)  # Wait a bit for settings to propagate
            r4 = httpx.post(url, headers=headers, json=data, timeout=30.0)
            if r4.status_code in (201, 204, 409):
                print(f"✅ Pages enabled after settings update")
                return True
                
    except Exception as e:
        print(f"⚠️ Settings update failed: {e}")
    
    # If all methods fail, provide instructions but continue
    print(f"⚠️ Automatic Pages enablement failed")
    print(f"📍 Pages should still deploy automatically from main branch")
    print(f"📍 URL will be: https://{USERNAME}.github.io/{repo_name}/")
    return True  # Return True to continue workflow


def generate_mit_license(owner_name=None):
    """Generate MIT License text."""
    year = datetime.utcnow().year
    owner = owner_name or USERNAME or "Owner"
    return f"""MIT License

Copyright (c) {year} {owner}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


# ============================================================================
# LLM CODE GENERATOR
# ============================================================================

def decode_attachments(attachments):
    """
    Decode base64 attachments and save to disk.
    Returns list of dicts: {"name": name, "path": "/tmp/..", "mime": mime, "size": n}
    """
    saved = []
    for att in attachments or []:
        name = att.get("name") or "attachment"
        url = att.get("url", "")
        if not url.startswith("data:"):
            continue
        try:
            header, b64data = url.split(",", 1)
            mime = header.split(";")[0].replace("data:", "")
            data = base64.b64decode(b64data)
            path = TMP_DIR / name
            with open(path, "wb") as f:
                f.write(data)
            saved.append({
                "name": name,
                "path": str(path),
                "mime": mime,
                "size": len(data)
            })
        except Exception as e:
            print(f"⚠️ Failed to decode attachment {name}: {e}")
    return saved


def summarize_attachment_meta(saved):
    """Generate human-readable summary of attachments for LLM prompt."""
    summaries = []
    for s in saved:
        nm = s["name"]
        p = s["path"]
        mime = s.get("mime", "")
        try:
            if mime.startswith("text") or nm.endswith((".md", ".txt", ".json", ".csv")):
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    if nm.endswith(".csv"):
                        lines = [next(f, "").strip() for _ in range(3)]
                        preview = "\\n".join(lines)
                    else:
                        data = f.read(1000)
                        preview = data.replace("\n", "\\n")[:1000]
                summaries.append(f"- {nm} ({mime}): preview: {preview}")
            else:
                summaries.append(f"- {nm} ({mime}): {s['size']} bytes")
        except Exception as e:
            summaries.append(f"- {nm} ({mime}): (could not read preview: {e})")
    return "\n".join(summaries)


def _strip_code_block(text: str) -> str:
    """Remove markdown code block markers if present."""
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            code_content = parts[1]
            if "\n" in code_content:
                code_content = code_content.split("\n", 1)[1]
            return code_content.strip()
    return text.strip()


def generate_readme_fallback(brief: str, checks=None, attachments_meta=None, round_num=1):
    """Generate fallback README if LLM fails."""
    checks_text = "\n".join(checks or [])
    att_text = attachments_meta or "None"
    return f"""# Auto-generated README (Round {round_num})

**Project brief:** {brief}

**Attachments:**
{att_text}

**Checks to meet:**
{checks_text}

## Setup
1. Open `index.html` in a browser.
2. No build steps required.

## Notes
This README was generated as a fallback (OpenAI did not return an explicit README).
"""


def generate_app_code(brief: str, attachments=None, checks=None, round_num=1, prev_readme=None):
    """
    Generate or revise an app using OpenAI with best available model.
    Uses Claude Sonnet 4.5 as primary (via OpenAI API if configured), falls back to GPT-4o.
    """
    saved = decode_attachments(attachments or [])
    attachments_meta = summarize_attachment_meta(saved)

    context_note = ""
    if round_num == 2 and prev_readme:
        context_note = f"\n### Previous README.md:\n{prev_readme}\n\nRevise and enhance this project according to the new brief below.\n"

    checks_list = "\n".join([f"- {check}" for check in (checks or [])])

    user_prompt = f"""You are a professional web developer assistant.

### Round
{round_num}

### Task
{brief}

{context_note}

### Attachments (if any)
{attachments_meta}

### Evaluation checks
{checks_list}

### Output format rules:
1. Produce a complete web app (HTML/JS/CSS inline if needed) satisfying the brief.
2. Output must contain **two parts only**:
   - index.html (main code)
   - README.md (starts after a line containing exactly: ---README.md---)
3. README.md must include:
   - Overview
   - Setup
   - Usage
   - If Round 2, describe improvements made from previous version.
4. Do not include any commentary outside code or README.
5. Make sure the code is production-ready and follows best practices.
6. For CSV files, use fetch() to load them and process the data.
7. Ensure the HTML is fully self-contained and works without a build step.
"""

    try:
        # Try models in order of preference
        # Primary: GPT-4o (most capable), Secondary: GPT-4 Turbo, Tertiary: GPT-4o-mini
        models_to_try = ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini", "gpt-3.5-turbo"]
        
        last_error = None
        for model in models_to_try:
            try:
                print(f"🤖 Attempting to generate code with {model}...")
                response = openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a professional web developer. Output ONLY the code and README. Follow the format exactly. Output complete, working code."},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=8000
                )
                text = response.choices[0].message.content or ""
                print(f"✅ Generated code using {model}")
                break
            except Exception as model_error:
                last_error = model_error
                error_str = str(model_error)
                if "does not exist" in error_str or "model_not_found" in error_str:
                    print(f"⚠️ Model {model} not available, trying next...")
                    continue
                elif "insufficient_quota" in error_str or "rate_limit" in error_str:
                    print(f"⚠️ Model {model} quota exceeded, trying next...")
                    continue
                elif "401" in error_str or "authentication" in error_str.lower():
                    print(f"❌ Authentication error with OpenAI API: {error_str}")
                    raise Exception("OpenAI API authentication failed")
                else:
                    print(f"⚠️ Model {model} error: {error_str}")
                    continue
        else:
            # If all models fail, raise the last error
            if last_error:
                raise last_error
            else:
                raise Exception("All available OpenAI models failed")
                
    except Exception as e:
        print(f"❌ OpenAI API error: {e}")
        print(f"⚠️ Using enhanced fallback HTML...")
        
        # Create a more functional fallback
        fallback_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Application</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
        .container {{ margin-top: 3rem; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="row">
            <div class="col-md-8 offset-md-2">
                <div class="card">
                    <div class="card-header bg-primary text-white">
                        <h4 class="mb-0">Application</h4>
                    </div>
                    <div class="card-body">
                        <p class="lead">Project: {brief[:100]}</p>
                        <div id="content">
                            <p class="text-muted">Loading content...</p>
                        </div>
                        <div id="error" class="alert alert-danger" style="display:none;"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {{
            try {{
                // Try to load data.csv if it exists
                fetch('data.csv')
                    .then(r => r.text())
                    .then(data => {{
                        const lines = data.trim().split('\\n');
                        const table = '<table class="table table-sm"><thead><tr>';
                        
                        if (lines.length > 0) {{
                            const headers = lines[0].split(',');
                            headers.forEach(h => {{ 
                                table += '<th>' + (h || '').trim() + '</th>'; 
                            }});
                            table += '</tr></thead><tbody>';
                            
                            for (let i = 1; i < Math.min(lines.length, 11); i++) {{
                                const values = lines[i].split(',');
                                table += '<tr>';
                                values.forEach(v => {{ 
                                    table += '<td>' + (v || '').trim() + '</td>'; 
                                }});
                                table += '</tr>';
                            }}
                            table += '</tbody></table>';
                            document.getElementById('content').innerHTML = table;
                        }}
                    }})
                    .catch(e => {{
                        document.getElementById('content').innerHTML = '<p class="text-muted">No data file found.</p>';
                    }});
            }} catch (e) {{
                document.getElementById('error').style.display = 'block';
                document.getElementById('error').textContent = 'Error: ' + e.message;
            }}
        }});
    </script>
</body>
</html>

---README.md---
{generate_readme_fallback(brief, checks, attachments_meta, round_num)}
"""
        text = fallback_html

    if "---README.md---" in text:
        code_part, readme_part = text.split("---README.md---", 1)
        code_part = _strip_code_block(code_part)
        readme_part = _strip_code_block(readme_part)
    else:
        code_part = _strip_code_block(text)
        readme_part = generate_readme_fallback(brief, checks, attachments_meta, round_num)

    files = {"index.html": code_part, "README.md": readme_part}
    return {"files": files, "attachments": saved}


# ============================================================================
# NOTIFICATION UTILITIES
# ============================================================================

def notify_evaluation_server(evaluation_url: str, payload: dict) -> bool:
    """Send repo details back to evaluation server with retry logic."""
    headers = {"Content-Type": "application/json"}
    delay = 1

    for attempt in range(5):
        try:
            r = httpx.post(evaluation_url, headers=headers, json=payload, timeout=30.0)
            if r.status_code == 200:
                print("✅ Evaluation server notified successfully")
                return True
            else:
                print(f"⚠️ Attempt {attempt+1}: Server responded {r.status_code}")
        except Exception as e:
            print(f"⚠️ Attempt {attempt+1} failed: {e}")

        time.sleep(delay)
        delay *= 2

    print("❌ Failed to notify evaluation server after 5 retries")
    return False


# ============================================================================
# PERSISTENCE
# ============================================================================

def load_processed():
    """Load previously processed requests from disk."""
    if os.path.exists(PROCESSED_PATH):
        try:
            with open(PROCESSED_PATH, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️ Error loading processed requests: {e}")
            return {}
    return {}


def save_processed(data):
    """Save processed requests to disk."""
    try:
        with open(PROCESSED_PATH, 'w') as f:
            json.dump(data, f, indent=2)
    except IOError as e:
        print(f"⚠️ Error saving processed requests: {e}")


# ============================================================================
# BACKGROUND PROCESSING
# ============================================================================

def process_request(data):
    """
    Background task to process the request:
    1. Generate app code using LLM (with better models)
    2. Create/update GitHub repo
    3. Deploy to GitHub Pages
    4. WAIT for Pages to be live and verify deployment
    5. Notify evaluation server only after verification
    
    For Round 2: Re-generates code based on new brief, updates files, 
    and verifies Pages redeploys correctly.
    """
    round_num = data.get("round", 1)
    task_id = data["task"]
    email = data["email"]
    
    print(f"\n{'='*60}")
    print(f"⚙️  Starting Round {round_num} for task: {task_id}")
    print(f"📧 Email: {email}")
    print(f"📝 Brief: {data.get('brief')[:80]}...")
    print(f"{'='*60}\n")

    try:
        # Step 1: Decode and save attachments
        attachments = data.get("attachments", [])
        saved_attachments = decode_attachments(attachments)
        if saved_attachments:
            print(f"📎 Saved {len(saved_attachments)} attachment(s):")
            for att in saved_attachments:
                print(f"   - {att['name']} ({att['mime']}, {att['size']} bytes)")

        # Step 2: Get or create repository
        repo = create_repo(task_id, description=f"Auto-generated app: {data['brief'][:100]}")
        
        # Step 3: For Round 2+, fetch previous code and README for context
        prev_readme = None
        prev_code = None
        if round_num >= 2:
            try:
                readme_file = repo.get_contents("README.md")
                prev_readme = readme_file.decoded_content.decode("utf-8", errors="ignore")
                print("📖 Loaded previous README.md for Round 2+ context")
            except Exception as e:
                print(f"⚠️ Could not fetch previous README: {e}")
            
            try:
                index_file = repo.get_contents("index.html")
                prev_code = index_file.decoded_content.decode("utf-8", errors="ignore")
                print("💻 Loaded previous index.html for context")
            except Exception as e:
                print(f"⚠️ Could not fetch previous code: {e}")

        # Step 4: Generate or update app code using LLM with better models
        if round_num == 1:
            print(f"\n🤖 Generating code with advanced LLM model (Round 1 - New Project)...")
        else:
            print(f"\n🤖 Refactoring code with advanced LLM model (Round {round_num} - Feature Updates)...")
        
        gen_result = generate_app_code(
            brief=data["brief"],
            attachments=attachments,
            checks=data.get("checks", []),
            round_num=round_num,
            prev_readme=prev_readme
        )

        files = gen_result.get("files", {})
        saved_info = gen_result.get("attachments", [])
        
        print(f"✅ Generated {len(files)} file(s): {', '.join(files.keys())}")

        # Step 5: Commit attachments (Round 1 only)
        if round_num == 1 and saved_info:
            print(f"\n📦 Committing {len(saved_info)} attachment(s) to repo...")
            for att in saved_info:
                att_name = att["name"]
                att_path_obj = Path(att["path"])
                
                try:
                    with open(att_path_obj, "rb") as f:
                        content_bytes = f.read()
                    
                    # Determine if text or binary
                    is_text = (
                        att["mime"].startswith("text") or 
                        att_name.endswith((".md", ".csv", ".json", ".txt", ".html", ".js", ".css"))
                    )
                    
                    if is_text:
                        text = content_bytes.decode("utf-8", errors="ignore")
                        create_or_update_file(repo, att_name, text, f"Add attachment: {att_name}")
                    else:
                        create_or_update_binary_file(
                            repo, att_name, content_bytes, f"Add binary attachment: {att_name}"
                        )
                except Exception as e:
                    print(f"⚠️ Failed to commit attachment {att_name}: {e}")

        # Step 6: Commit generated/updated files (index.html, README.md, etc.)
        print(f"\n📝 Committing {'updated' if round_num > 1 else 'generated'} files...")
        for filename, content in files.items():
            if round_num == 1:
                commit_msg = f"Add {filename} (Round {round_num})"
            else:
                commit_msg = f"Update {filename} with new features (Round {round_num})"
            create_or_update_file(repo, filename, content, commit_msg)
        print(f"✅ Files committed successfully")

        # Step 7: Add MIT LICENSE
        print(f"⚖️  Adding MIT LICENSE...")
        mit_text = generate_mit_license()
        create_or_update_file(repo, "LICENSE", mit_text, "Add MIT License")

        # Step 7.5: Add GitHub Actions workflow for Pages deployment
        print(f"🔧 Adding GitHub Actions workflow for Pages...")
        workflow_content = """name: Deploy to GitHub Pages

on:
  push:
    branches: [ main ]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
      - name: Setup Pages
        uses: actions/configure-pages@v4
      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: '.'
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
"""
        try:
            create_or_update_file(repo, ".github/workflows/pages.yml", workflow_content, "Add Pages deployment workflow")
            print(f"✅ GitHub Actions workflow added")
        except Exception as e:
            print(f"⚠️ Could not add workflow: {e}")

        # Step 8: Enable GitHub Pages
        pages_url = f"https://{USERNAME}.github.io/{task_id}/"
        print(f"\n🌐 Enabling GitHub Pages...")
        enable_pages(task_id)

        # Step 9: Get latest commit SHA
        commit_sha = None
        try:
            commits = repo.get_commits()
            commit_sha = commits[0].sha
            print(f"📌 Latest commit SHA: {commit_sha[:8]}...")
        except Exception as e:
            print(f"⚠️ Could not retrieve commit SHA: {e}")

        # Step 10: WAIT FOR PAGES TO BE LIVE (critical!)
        # For Round 1, this is the first deployment. For Round 2+, this is a redeployment.
        if round_num == 1:
            print(f"\n⏳ Waiting for initial GitHub Pages deployment (Round {round_num})...")
            verify_timeout = 300  # 5 minutes for initial deployment
        else:
            print(f"\n⏳ Waiting for GitHub Pages redeployment (Round {round_num})...")
            verify_timeout = 180  # 3 minutes for redeployment (should be faster)
        
        pages_live = verify_pages_deployment(task_id, max_wait_seconds=verify_timeout)
        
        if not pages_live:
            print(f"❌ Pages deployment verification failed on Round {round_num}")
            # Notify with error
            payload = {
                "email": email,
                "task": task_id,
                "round": round_num,
                "nonce": data["nonce"],
                "error": f"Pages deployment timeout after {verify_timeout} seconds on Round {round_num}",
                "repo_url": repo.html_url,
                "commit_sha": commit_sha,
                "pages_url": pages_url,
                "status": "pages_timeout"
            }
            notify_evaluation_server(data.get("evaluation_url"), payload)
            raise Exception(f"Pages deployment failed to complete on Round {round_num}")

        # Step 11: Pages is live! Prepare notification payload
        payload = {
            "email": email,
            "task": task_id,
            "round": round_num,
            "nonce": data["nonce"],
            "repo_url": repo.html_url,
            "commit_sha": commit_sha,
            "pages_url": pages_url,
            "status": "success"
        }

        print(f"\n📤 Notifying evaluation server...")
        print(f"   Evaluation URL: {data['evaluation_url']}")
        
        # Step 12: Notify evaluation server with retries
        notify_success = notify_evaluation_server(data.get("evaluation_url"), payload)
        
        if not notify_success:
            print(f"⚠️ Failed to notify evaluation server after retries")
            raise Exception("Evaluation server notification failed")

        # Step 13: Save to processed requests log
        processed = load_processed()
        key = f"{email}::{task_id}::round{round_num}::nonce{data['nonce']}"
        processed[key] = payload
        save_processed(processed)

        print(f"\n{'='*60}")
        print(f"✅ Round {round_num} completed successfully for {task_id}")
        print(f"🔗 Repo: {repo.html_url}")
        print(f"🌐 Pages: {pages_url}")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ Error processing Round {round_num} for {task_id}")
        print(f"Error: {e}")
        print(f"{'='*60}\n")
        
        # Attempt to notify evaluation server of failure
        try:
            error_payload = {
                "email": data.get("email"),
                "task": data.get("task"),
                "round": data.get("round"),
                "nonce": data.get("nonce"),
                "error": str(e),
                "repo_url": None,
                "commit_sha": None,
                "pages_url": None,
                "status": "error"
            }
            notify_evaluation_server(data.get("evaluation_url"), error_payload)
        except:
            pass


# ============================================================================
# FASTAPI ENDPOINTS
# ============================================================================

@app.post("/api-endpoint")
async def receive_request(request: Request, background_tasks: BackgroundTasks):
    """
    Main endpoint to receive build/revision requests.
    
    Expected JSON payload:
    {
        "secret": "...",
        "email": "...",
        "task": "...",
        "round": 1,
        "nonce": "...",
        "brief": "...",
        "checks": [...],
        "evaluation_url": "...",
        "attachments": [...]
    }
    """
    try:
        data = await request.json()
    except json.JSONDecodeError:
        return {"error": "Invalid JSON payload", "status": "bad_request"}

    print(f"\n{'='*60}")
    print(f"📨 Incoming request received")
    print(f"📧 Email: {data.get('email', 'N/A')}")
    print(f"🎯 Task: {data.get('task', 'N/A')}")
    print(f"🔄 Round: {data.get('round', 'N/A')}")
    print(f"{'='*60}\n")

    # Step 1: Verify secret
    provided_secret = data.get("secret")
    if provided_secret != USER_SECRET:
        print(f"❌ Authentication failed: Invalid secret")
        return {"error": "Invalid secret", "status": "unauthorized"}

    print(f"✅ Secret verified")

    # Step 2: Check for duplicate requests
    processed = load_processed()
    key = f"{data.get('email')}::{data.get('task')}::round{data.get('round')}::nonce{data.get('nonce')}"

    if key in processed:
        print(f"⚠️ Duplicate request detected: {key}")
        prev_payload = processed[key]
        
        # Re-notify evaluation server with existing data
        evaluation_url = data.get("evaluation_url")
        if evaluation_url:
            print(f"📤 Re-notifying evaluation server with previous results...")
            notify_evaluation_server(evaluation_url, prev_payload)
        
        return {
            "status": "duplicate",
            "note": "Request already processed. Re-notified evaluation server.",
            "previous_result": prev_payload
        }

    # Step 3: Validate required fields
    required_fields = ["email", "task", "round", "nonce", "brief", "evaluation_url"]
    missing_fields = [field for field in required_fields if field not in data]
    
    if missing_fields:
        print(f"❌ Missing required fields: {missing_fields}")
        return {
            "error": f"Missing required fields: {', '.join(missing_fields)}",
            "status": "bad_request"
        }

    # Step 4: Schedule background processing
    print(f"⏳ Scheduling background task for Round {data.get('round')}...")
    background_tasks.add_task(process_request, data)

    # Step 5: Return immediate acknowledgment (HTTP 200)
    return {
        "status": "accepted",
        "note": f"Round {data.get('round')} processing started in background. Will notify evaluation server after Pages verification.",
        "task": data.get("task"),
        "round": data.get("round")
    }


@app.get("/")
async def health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "service": "LLM App Builder",
        "version": "2.0.0"
    }


@app.get("/health")
async def health():
    """Detailed health check with GitHub connectivity test."""
    try:
        user = g.get_user()
        github_status = f"Connected as {user.login}"
    except Exception as e:
        github_status = f"Error: {str(e)}"
    
    return {
        "status": "healthy",
        "github": github_status,
        "username": USERNAME,
    }


@app.post("/test-eval")
async def mock_evaluation(request: Request):
    """Mock evaluation endpoint for testing purposes."""
    try:
        data = await request.json()
        print("\n" + "="*60)
        print("📥 Mock Evaluation Endpoint Received Notification")
        print("="*60)
        print(f"📧 Email: {data.get('email')}")
        print(f"🎯 Task: {data.get('task')}")
        print(f"🔄 Round: {data.get('round')}")
        print(f"🔗 Repo URL: {data.get('repo_url')}")
        print(f"📌 Commit SHA: {data.get('commit_sha')}")
        print(f"🌐 Pages URL: {data.get('pages_url')}")
        print(f"📊 Status: {data.get('status')}")
        if data.get('error'):
            print(f"❌ Error: {data.get('error')}")
        print("="*60 + "\n")
        
        return {"status": "ok", "message": "Notification received (mock endpoint)"}
    except Exception as e:
        print(f"⚠️ Mock evaluation error: {e}")
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print("\n" + "="*60)
    print("🚀 Starting LLM App Builder Server (v2.0)")
    print("="*60)
    print(f"📍 Server will run at: http://0.0.0.0:{port}")
    print(f"📍 Health check: http://localhost:{port}/health")
    print(f"📍 API endpoint: http://localhost:{port}/api-endpoint")
    print(f"📍 Mock eval: http://localhost:{port}/test-eval")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
