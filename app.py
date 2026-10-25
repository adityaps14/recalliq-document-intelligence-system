
from pathlib import Path
import re

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
)
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"

UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}
MAX_FILE_SIZE = 8 * 1024 * 1024  # 8 MB

app = Flask(__name__)

# For a local student project. Use an environment variable
# for SECRET_KEY before deploying publicly.
app.config["SECRET_KEY"] = "change-this-for-your-own-local-project"
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE


# --------------------------------------------------
# Document extraction
# --------------------------------------------------

def read_document(path: Path) -> str:
    """Extract text from TXT, Markdown, or text-based PDF files."""

    try:
        if path.suffix.lower() == ".pdf":
            reader = PdfReader(str(path))
            pages = [
                page.extract_text() or ""
                for page in reader.pages
            ]
            return "\n".join(pages)

        return path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

    except Exception:
        app.logger.exception("Could not read document: %s", path.name)
        return ""


def load_documents():
    """Load supported documents from the local uploads directory."""

    documents = []

    for path in sorted(UPLOAD_DIR.iterdir()):
        if not path.is_file():
            continue

        if path.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue

        text = read_document(path).strip()

        if text:
            documents.append({
                "name": path.name,
                "text": text,
            })

    return documents


# --------------------------------------------------
# Passage splitting and document search
# --------------------------------------------------

def split_passages(text, words_per_passage=120):
    """Split a document into smaller passages for searching."""

    words = text.split()

    passages = [
        " ".join(words[index:index + words_per_passage])
        for index in range(0, len(words), words_per_passage)
    ]

    return passages or [text]


def search_documents(query, docs):
    """Rank document passages using TF-IDF and cosine similarity."""

    if not query.strip() or not docs:
        return []

    passages = []

    for document in docs:
        for passage in split_passages(document["text"]):
            passages.append({
                "name": document["name"],
                "text": passage,
            })

    if not passages:
        return []

    corpus = [item["text"] for item in passages] + [query]

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(corpus)

        scores = cosine_similarity(
            matrix[-1],
            matrix[:-1],
        ).flatten()

    except ValueError:
        # For example, the query may contain only stop words.
        return []

    ranked = sorted(
        zip(scores, passages),
        key=lambda item: item[0],
        reverse=True,
    )

    return [
        {
            **passage,
            "score": round(float(score), 3),
        }
        for score, passage in ranked[:5]
        if score > 0
    ]


# --------------------------------------------------
# Extractive summarization
# --------------------------------------------------

def summarize(text, max_sentences=3):
    """
    Generate a short extractive summary.

    This selects useful sentences from the document.
    It does not generate new sentences using an LLM.
    """

    clean_text = re.sub(r"\s+", " ", text).strip()

    if not clean_text:
        return "No text available to summarize."

    # Split at sentence endings and common section headings.
    parts = re.split(
        r"(?<=[.!?])\s+"
        r"|\s+(?=(?:AIM|OBJECTIVES?|LEARNING OUTCOMES?|"
        r"CONCLUSION|PURPOSE)\b)",
        clean_text,
        flags=re.IGNORECASE,
    )

    useful = []

    metadata_labels = (
        "student name:",
        "uid:",
        "section/group:",
        "date of performance:",
        "subject code:",
    )

    for part in parts:
        sentence = part.strip(" \t\r\n•-")
        words = sentence.split()
        lower = sentence.lower()

        if len(words) < 5:
            continue

        # Skip sentences that are primarily student metadata.
        if any(label in lower for label in metadata_labels):
            continue

        # Skip obvious long code fragments.
        if sentence.count(";") >= 3 or "{" in sentence:
            continue

        useful.append(sentence)

    if not useful:
        return (
            clean_text[:297] + "..."
            if len(clean_text) > 300
            else clean_text
        )

    keywords = (
        "aim",
        "objective",
        "purpose",
        "calculate",
        "function",
        "discount",
        "final amount",
        "learning outcomes",
        "understand",
        "display",
        "customer",
        "conclusion",
    )

    def score(sentence):
        lower = sentence.lower()

        points = sum(
            2 for keyword in keywords
            if keyword in lower
        )

        word_count = len(sentence.split())

        # Prefer concise, readable sentences.
        if word_count > 45:
            points -= 2

        if word_count > 70:
            points -= 4

        return points

    # Prefer informative sentences; preserve original order
    # when presenting the final summary.
    ranked = sorted(
        enumerate(useful),
        key=lambda item: (-score(item[1]), item[0]),
    )

    selected = sorted(
        ranked[:max(1, max_sentences)],
        key=lambda item: item[0],
    )

    summary = " ".join(
        sentence for _, sentence in selected
    )

    # Prevent an excessively long summary.
    if len(summary) > 700:
        summary = summary[:697].rsplit(" ", 1)[0] + "..."

    return summary


# --------------------------------------------------
# Main route
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    query = ""
    summary = ""

    docs = load_documents()

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        action = request.form.get("action", "search")

        if action == "upload":
            file = request.files.get("document")

            if not file or not file.filename:
                flash(
                    "Choose a TXT, MD, or text-based PDF file.",
                    "error",
                )

            else:
                safe_name = Path(file.filename).name
                suffix = Path(safe_name).suffix.lower()

                if suffix not in ALLOWED_EXTENSIONS:
                    flash(
                        "Unsupported file type. Use .txt, .md, or .pdf.",
                        "error",
                    )

                else:
                    destination = UPLOAD_DIR / safe_name

                    # Avoid overwriting an existing document.
                    if destination.exists():
                        flash(
                            "A file with this name already exists. "
                            "Rename your file and try again.",
                            "error",
                        )

                    else:
                        file.save(destination)

                        flash(
                            f"Added {safe_name} to your local document library.",
                            "success",
                        )

                        return redirect(url_for("index"))

        elif action == "summarize":
            selected_name = request.form.get(
                "document_name",
                "",
            )

            match = next(
                (
                    document for document in docs
                    if document["name"] == selected_name
                ),
                None,
            )

            if match:
                summary = summarize(match["text"])

            else:
                flash(
                    "Select a document to summarize.",
                    "error",
                )

        else:
            results = search_documents(query, docs)

            if query and not results:
                flash(
                    "No matching passages found. Try different keywords.",
                    "info",
                )

    return render_template(
        "index.html",
        docs=docs,
        results=results,
        query=query,
        summary=summary,
    )


# --------------------------------------------------
# Document deletion
# --------------------------------------------------

@app.route("/delete/<path:name>", methods=["POST"])
def delete_document(name):
    target = (UPLOAD_DIR / Path(name).name).resolve()

    # Only delete files directly inside the uploads directory.
    if (
        target.parent == UPLOAD_DIR.resolve()
        and target.is_file()
        and target.suffix.lower() in ALLOWED_EXTENSIONS
    ):
        target.unlink()

        flash(
            f"Removed {target.name} from the local library.",
            "success",
        )

    return redirect(url_for("index"))


# --------------------------------------------------
# Application entry point
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)