# Security & Release Checklist

> Before releasing the code publicly (open-source or anonymized submission), run through this checklist.

## Pre-Release Security Scan

### 1. API Keys and Secrets

```bash
# Scan for hardcoded API keys
grep -rn "API_KEY\|api_key.*=\|sk-\|ZHIPU\|OPENAI" src/ --include="*.py"

# Verify .env is in .gitignore
grep "\.env" .gitignore
```

- [ ] No hardcoded API keys in source code
- [ ] `.env` is in `.gitignore`
- [ ] `.env.example` exists with placeholder values only
- [ ] All API keys are loaded via `os.environ.get()` or `python-dotenv`

### 2. Private Paths

```bash
# Scan for absolute paths that may leak information
grep -rn "/home/\|/root/\|/Users/\|C:\\" src/ --include="*.py"
```

- [ ] No absolute paths referencing private directories
- [ ] All paths are relative to project root or from environment variables

### 3. Hardcoded Results

- [ ] No manually written metrics/tables in source code
- [ ] All results come from `src_result/eval/*.json` files
- [ ] No "magic numbers" that appear to be paper results

### 4. Local Data and Checkpoints

```bash
# Check for leaked data files
ls data/
ls src_result/
```

- [ ] No raw dataset files in repo (only parquet; covered by .gitignore)
- [ ] No trained model checkpoints (`.pth`) in repo
- [ ] No local evaluation logs that may contain private info

### 5. Documentation

- [ ] README.md has no sensitive information
- [ ] No personal names or emails in comments
- [ ] No internal project names or team references
- [ ] `docs/` folder reviewed for private content

### 6. Test Files

```bash
# Ensure tests don't contain real API keys
grep -rn "sk-\|api_key\|secret" tests/ --include="*.py"
```

- [ ] Tests use mock/environment-variable keys
- [ ] No real credentials in test fixtures

---

## Anonymized Submission Checklist

If submitting to an anonymized venue:

- [ ] No author names in README.md
- [ ] No GitHub username / author info in `git log`
- [ ] Consider: `git filter-branch` or BFG to remove author history if needed
- [ ] No commit messages revealing author identity
- [ ] No file paths that reference personal directories

---

## BFG Repo-Cleaner (for removing secrets from history)

```bash
# Install BFG
brew install bfg  # macOS
# or: wget https://repo1.maven.org/maven2/com/madgag/bfg/1.14.12/bfg-1.14.12.jar

# Remove API keys from history
bfg --replace-text private_keys.txt --no-blob-protection repo.git

# Remove large files
bfg --strip-blobs-bigger-than 100M repo.git

# Force push (after verification)
git reflog expire --expire=now --all && git gc --prune=now --aggressive
git push --force
```

---

## Verification Commands

Run these before every release:

```bash
# 1. No API keys
! grep -rn "ad443e\|5f2e493\|5f2e493b563348e28d9748fe0020f760" .

# 2. No .env file tracked
! git ls-files | grep "\.env$"

# 3. README has no personal info
! grep -i "author\|contact\|email" README.md

# 4. Tests pass
pytest tests/ -q

# 5. Smoke test runs
python -m src.llm_clp.training.train_causal_fair --help
```