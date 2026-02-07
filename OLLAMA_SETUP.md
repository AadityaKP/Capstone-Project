# Ollama Setup Guide (Windows)

Follow these steps to set up the Llama 3.1:8b model locally.

## 1. Install Ollama
1. Download the Windows installer from [ollama.com/download](https://ollama.com/download).
2. Run the installer.
3. **Restart your terminal** (PowerShell or Command Prompt).

## 2. Verify Installation
Run the following command in your terminal:
```powershell
ollama --version
```
*If it prints a version number, Ollama is installed.*

## 3. Download the Model
Pull the required model (approx. 4-5 GB):
```powershell
ollama pull llama3.1:8b
```

## 4. Sanity Check
Test the model interactively:
```powershell
ollama run llama3.1:8b
```
- Type `hello`.
- If it replies, it works.
- Type `/exit` to quit.

## 5. Python Setup
Install the Python library:
```powershell
pip install ollama
```

## Notes
- **Server**: Ollama automatically runs a local server at `http://localhost:11434`. You do not need to start it manually.
- **Hardware**: Works best with 16GB+ RAM. 
