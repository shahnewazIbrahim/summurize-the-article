import argparse
import textwrap

import requests
from bs4 import BeautifulSoup
from newspaper import Article
from transformers import pipeline
from tqdm import tqdm


# ---------------------------
# Article extraction helpers
# ---------------------------

def extract_with_newspaper(url: str) -> str:
    """
    Try to extract article text using newspaper3k.
    """
    article = Article(url)
    article.download()
    article.parse()
    text = (article.text or "").strip()
    return text


def extract_with_bs4(url: str) -> str:
    """
    Fallback method: use requests + BeautifulSoup
    to scrape visible text from the page.
    """
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    html = resp.text

    soup = BeautifulSoup(html, "html.parser")

    # Remove common noise
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form"]):
        tag.decompose()

    # Combine paragraphs
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = "\n".join(p for p in paragraphs if len(p.split()) > 3)
    return text.strip()


def get_article_text(url: str) -> str:
    """
    Try multiple methods to get clean article text.
    """
    print(f"[i] Fetching article from: {url}")

    # 1) Try newspaper3k
    try:
        text = extract_with_newspaper(url)
        if len(text.split()) > 100:
            print("[i] Extracted text using newspaper3k.")
            return text
        else:
            print("[!] newspaper3k text too short, trying BeautifulSoup fallback...")
    except Exception as e:
        print(f"[!] newspaper3k failed: {e}. Using BeautifulSoup fallback...")

    # 2) Fallback: BeautifulSoup
    try:
        text = extract_with_bs4(url)
        if len(text.split()) > 30:
            print("[i] Extracted text using BeautifulSoup.")
            return text
    except Exception as e:
        print(f"[!] BeautifulSoup fallback failed: {e}")

    raise RuntimeError("Could not extract a valid article from this URL.")


# ---------------------------
# Summarization helpers
# ---------------------------

def chunk_text(text: str, max_words: int = 600):
    """
    Long আর্টিকেলকে ছোট ছোট chunk এ ভাগ করি,
    যেন model input limit cross না করে।
    """
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i + max_words])


def build_summarizer(model_name: str = "facebook/bart-large-cnn"):
    """
    HuggingFace pipeline বানাই – summarization model।
    চাইলে model_name change করে অন্য model use করতে পারো।
    """
    print(f"[i] Loading summarizer model: {model_name}")
    summarizer = pipeline(
        "summarization",
        model=model_name,
        tokenizer=model_name,
    )
    return summarizer


def summarize_long_text(
    text: str,
    summarizer,
    chunk_size_words: int = 600,
    max_summary_words: int = 130,
    min_summary_words: int = 30,
) -> str:
    """
    বড় টেক্স্ট হলে chunk ধরে ধরে summarize করে পরে সবগুলো
    partial summary মিলে আবার একটা final summary করে।
    """
    chunks = list(chunk_text(text, max_words=chunk_size_words))

    if not chunks:
        raise ValueError("No text to summarize")

    print(f"[i] Total chunks: {len(chunks)}")

    partial_summaries = []

    for i, chunk in enumerate(tqdm(chunks, desc="Summarizing chunks")):
        # Transformer model এর max_length / min_length হলো token এ,
        # কিন্তু প্রায় আনুমানিক word count ধরে নিচ্ছি।
        summary = summarizer(
            chunk,
            max_length=max_summary_words,
            min_length=min_summary_words,
            do_sample=False,
        )[0]["summary_text"]

        partial_summaries.append(summary.strip())

    if len(partial_summaries) == 1:
        return partial_summaries[0]

    # Partial summaries মিলিয়ে আবার final summary
    combined = " ".join(partial_summaries)
    print("[i] Combining partial summaries into a final summary...")

    final_summary = summarizer(
        combined,
        max_length=max_summary_words,
        min_length=min_summary_words,
        do_sample=False,
    )[0]["summary_text"]

    return final_summary.strip()


# ---------------------------
# CLI interface
# ---------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="AI News Summarizer – URL থেকে নিউজ পড়ে ছোট summary বানায়।"
    )
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="News article এর URL",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/bart-large-cnn",
        help="HuggingFace summarization model name (default: facebook/bart-large-cnn)",
    )
    parser.add_argument(
        "--max-summary-words",
        type=int,
        default=150,
        help="Summary এর সর্বোচ্চ approx word সংখ্যা (model token limit অনুযায়ী)",
    )
    parser.add_argument(
        "--min-summary-words",
        type=int,
        default=40,
        help="Summary এর সর্বনিম্ন approx word সংখ্যা",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1) Article text আনো
    article_text = get_article_text(args.url)
    print()
    print("=" * 80)
    print("ORIGINAL ARTICLE (first 600 characters):")
    print("=" * 80)
    print(article_text[:600] + ("..." if len(article_text) > 600 else ""))
    print()

    # 2) Summarizer load
    summarizer = build_summarizer(args.model)

    # 3) Summarize
    summary = summarize_long_text(
        article_text,
        summarizer,
        max_summary_words=args.max_summary_words,
        min_summary_words=args.min_summary_words,
    )

    print()
    print("=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print(textwrap.fill(summary, width=100))
    print()


if __name__ == "__main__":
    main()
