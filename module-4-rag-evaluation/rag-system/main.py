"""RAG System - FastAPI Application."""
import os
import shutil
import tempfile
import urllib.parse
import urllib.request
import zipfile
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.llm_client import get_llm_client
from src import CodebaseRAG, RAGEvaluator, create_eval_dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(
    title="Codebase RAG System",
    description="RAG system for querying codebases with evaluation",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for UI
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class QueryRequest(BaseModel):
    """Request for querying the codebase."""
    question: str
    n_results: int = 5
    filter_language: Optional[str] = None


class IndexDirectoryRequest(BaseModel):
    """Request to index a directory."""
    directory: str
    extensions: Optional[List[str]] = None


class IndexGitHubRequest(BaseModel):
    """Request to index a GitHub repository."""
    url: str
    branch: Optional[str] = None
    extensions: Optional[List[str]] = None


class IndexFilesRequest(BaseModel):
    """Request to index files directly."""
    files: Dict[str, str]  # filename -> content


class EvalRequest(BaseModel):
    """Request for evaluation."""
    examples: List[Dict[str, Any]]


class QueryResponse(BaseModel):
    """Response from query."""
    answer: str
    sources: List[Dict[str, Any]]
    context_used: str


DEFAULT_EXTENSIONS = [
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go",
    ".cs", ".cshtml", ".csproj", ".sln", ".razor",
    ".rb", ".php", ".rs", ".kt", ".swift", ".scala",
    ".c", ".cpp", ".h", ".hpp",
    ".html", ".css", ".scss", ".less",
    ".sql", ".sh", ".bash", ".ps1",
    ".json", ".yaml", ".yml", ".toml", ".xml",
    ".md", ".txt",
    ".dockerfile", ".tf", ".hcl",
]
IGNORED_DIRECTORIES = [
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    ".next",
]


def _count_indexable_files(base_path: str, extensions: Optional[List[str]] = None) -> int:
    """Count code files that match indexing rules."""
    exts = extensions or DEFAULT_EXTENSIONS
    files_count = 0

    for root, dirs, files in os.walk(base_path):
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRECTORIES]
        files_count += sum(1 for file in files if any(file.endswith(ext) for ext in exts))

    return files_count


def _parse_github_url(url: str) -> Dict[str, Optional[str]]:
    """Parse GitHub URL and extract owner/repo/optional branch."""
    parsed = urllib.parse.urlparse(url.strip())

    if parsed.scheme not in ("http", "https"):
        raise ValueError("GitHub URL must start with http:// or https://")

    if parsed.netloc.lower() not in ("github.com", "www.github.com"):
        raise ValueError("Only github.com URLs are supported")

    path_parts = [part for part in parsed.path.split("/") if part]
    if len(path_parts) < 2:
        raise ValueError("GitHub URL format: https://github.com/<owner>/<repo>")

    owner = path_parts[0]
    repo = path_parts[1]
    if repo.endswith(".git"):
        repo = repo[:-4]

    branch = None
    if len(path_parts) >= 4 and path_parts[2] == "tree":
        branch = path_parts[3]

    return {"owner": owner, "repo": repo, "branch": branch}


def _download_github_archive(owner: str, repo: str, preferred_branch: Optional[str], zip_path: str) -> str:
    """Download GitHub repo as zip; returns resolved branch."""
    branches_to_try = [preferred_branch] if preferred_branch else ["main", "master"]
    last_error = None

    for branch in branches_to_try:
        archive_url = f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/{branch}"
        try:
            with urllib.request.urlopen(archive_url, timeout=60) as response, open(zip_path, "wb") as out_file:
                shutil.copyfileobj(response, out_file)
            return branch
        except Exception as exc:
            last_error = exc

    raise ValueError(f"Could not download repository archive. Last error: {last_error}")


def _resolve_persist_directory() -> str:
    """Resolve a writable persistence directory across environments."""
    if os.getenv("VERCEL"):
        return "/tmp/chroma_db"

    return os.getenv("RAG_PERSIST_DIRECTORY", "./chroma_db")


# Initialize RAG
provider = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
llm = get_llm_client(provider)
persist_directory = _resolve_persist_directory()
os.makedirs(persist_directory, exist_ok=True)
rag = CodebaseRAG(llm, persist_directory=persist_directory)


@app.get("/")
async def root():
    """Serve the UI."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/index/directory")
async def index_directory(request: IndexDirectoryRequest):
    """Index a codebase directory."""
    try:
        files_count = _count_indexable_files(request.directory, request.extensions)
        count = rag.index_directory(request.directory, request.extensions)
        return {"indexed_chunks": count, "directory": request.directory, "files_processed": files_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index/github")
async def index_github_repository(request: IndexGitHubRequest):
    """Download and index a public GitHub repository by URL."""
    try:
        repo_info = _parse_github_url(request.url)
        branch = request.branch or repo_info["branch"]

        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, "repo.zip")
            resolved_branch = _download_github_archive(
                repo_info["owner"], repo_info["repo"], branch, zip_path
            )

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)

            extracted_dir = os.path.join(temp_dir, f"{repo_info['repo']}-{resolved_branch}")
            if not os.path.isdir(extracted_dir):
                extracted_candidates = [
                    name for name in os.listdir(temp_dir)
                    if os.path.isdir(os.path.join(temp_dir, name))
                ]
                if not extracted_candidates:
                    raise ValueError("Downloaded archive did not contain a repository directory")
                extracted_dir = os.path.join(temp_dir, extracted_candidates[0])

            files_count = _count_indexable_files(extracted_dir, request.extensions)
            count = rag.index_directory(extracted_dir, request.extensions)

        return {
            "indexed_chunks": count,
            "repository": f"{repo_info['owner']}/{repo_info['repo']}",
            "branch": resolved_branch,
            "files_processed": files_count,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index/files")
async def index_files(request: IndexFilesRequest):
    """Index files from request body."""
    try:
        count = rag.index_files(request.files)
        return {"indexed_chunks": count, "files": list(request.files.keys())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_codebase(request: QueryRequest):
    """Query the codebase."""
    try:
        result = rag.query(
            request.question,
            request.n_results,
            request.filter_language
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate")
async def evaluate_rag(request: EvalRequest):
    """Evaluate RAG performance."""
    try:
        examples = create_eval_dataset(request.examples)
        evaluator = RAGEvaluator(rag, llm)

        retrieval_metrics = evaluator.evaluate_retrieval(examples)
        generation_metrics = evaluator.evaluate_generation(examples)

        return {
            "retrieval": retrieval_metrics,
            "generation": generation_metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get index statistics."""
    return rag.get_stats()


@app.delete("/index")
async def clear_index():
    """Clear the index."""
    rag.clear_index()
    return {"status": "cleared"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "provider": provider}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
