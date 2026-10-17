# llm_sources — Multi-Source LLM Utility

A command-line tool that reads one or more sources (text, CSV, Word, PDF, HTML files and web pages), extracts their text and asks an LLM to summarize them or answer a question about them.

```
source → reader/parser → plain text → labelled, delimited prompt → LLM API → stdout or file
```

## Quick start on Windows (VS Code + PowerShell)

1. Unzip `llm-sources.zip` and open the `llm-sources` folder in VS Code (**File → Open Folder**).
2. Open a terminal with **Terminal → New Terminal**. It opens in PowerShell, already inside the project folder.
3. Install the libraries (once):
   ```
   python -m pip install -r requirements.txt
   ```
4. Set the key and server (each time you open a new terminal). For Gemini:
   ```
   $env:OPENAI_API_KEY="your-gemini-key"
   $env:OPENAI_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
   $env:LLM_MODEL="gemini-3.5-flash-lite"
   ```
5. Run it:
   ```
   python llm_sources.py samples/report.pdf
   ```

## Project structure

```
llm-sources/
├── llm_sources.py        the program
├── requirements.txt      Python libraries
├── README.md             this file
├── .gitignore
├── output/               place for results saved with -o
└── samples/              test files, one per supported format
    ├── notes.txt
    ├── sales.csv
    ├── BashTutorial.docx
    ├── report.pdf
    ├── article.html          contains a hidden prompt injection
    └── injection_test.txt    contains a visible prompt injection
```

## Setup

```
python -m pip install -r requirements.txt
```

Set the API key as an environment variable (never in the code):

```
$env:OPENAI_API_KEY="your-key"          # PowerShell
export OPENAI_API_KEY="your-key"        # macOS / Linux
```

Optional, to use any OpenAI-compatible server instead of OpenAI, for example Gemini:

```
$env:OPENAI_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
```

## Usage

```
python llm_sources.py samples/report.pdf
python llm_sources.py samples/report.pdf samples/BashTutorial.docx
python llm_sources.py samples/report.pdf https://example.com/article
python llm_sources.py samples/BashTutorial.docx -q "List all Bash commands found in this document."
python llm_sources.py samples/report.pdf -o output/summary.md
```

| Option | Description |
|---|---|
| `sources` | One or more files or HTTP/HTTPS URLs |
| `-q`, `--query` | Prompt for the LLM (default: "Summarize the supplied sources.") |
| `-o`, `--output FILE` | Save the result to a file instead of printing it |
| `-v`, `--verbose` | Show source types, sizes, token estimates and timings on stderr |
| `-m`, `--model` | Model name (default: `$LLM_MODEL` or `gpt-4o-mini`) |
| `--max-input-tokens` | Input budget per request; larger inputs are chunked (default 24000) |
| `--max-output-tokens` | Maximum tokens per answer (default 2000) |
| `--max-chunks` | Maximum chunked calls before the input is rejected (default 30) |
| `--show-prompt` | Only print the extracted text, without calling the LLM |

## How it works

1. **Type detection** — URLs by their HTTP `Content-Type` (HTML, PDF, CSV or plain text); files by extension (`.txt .md .csv .docx .pdf .html`).
2. **Extraction** — PyMuPDF for PDF (with page markers), python-docx for Word (headings and tables), the `csv` module for CSV, requests + BeautifulSoup for web pages.
3. **Labelling** — each source gets a label (`S1`, `S2`, …) and is wrapped in markers that contain a random string, so a document cannot fake the end of its own block. The answer cites these labels, and a source list is appended.
4. **Context window** — the prompt size is estimated with tiktoken (or ~4 characters per token as a fallback). If it exceeds `--max-input-tokens`, the sources are split into chunks, relevant notes are extracted from each chunk, and the final answer is written from the notes (map-reduce). If that would take more than `--max-chunks` calls, the program stops with an explanation.
5. **Errors** — missing files, unsupported formats and HTTP failures skip only that source with a warning; API errors (wrong key, quota, unknown model, timeouts, connection problems) stop with a clear message.

## Security

Every document and web page is treated as untrusted:

- The system prompt tells the model that source contents are data, never instructions, and that they must not override the task.
- HTML is cleaned before use: scripts, styles, comments, navigation, headers, footers, forms, iframes and hidden elements (`display:none`, `hidden`, `aria-hidden`) are removed. Hidden text is a common place to plant injections.
- Sources are delimited with per-request random markers, and the task is repeated after the sources.

`samples/injection_test.txt` and `samples/article.html` contain planted injections for testing.

## Sample files

`samples/` contains a text file, a CSV, a Word document, a two-page PDF and an HTML page.
