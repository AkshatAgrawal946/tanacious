# Launch the dashboard through Streamlit. Do not run app.py with `python app.py`.
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $python)) {
    throw "Virtual environment not found. Run: python -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
}

& $python -m streamlit run (Join-Path $root 'app.py') --server.headless false
