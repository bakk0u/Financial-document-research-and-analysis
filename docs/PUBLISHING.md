# Publish a standalone GitHub repository

The local ZIP contains source, tests, tagged model artifacts, frozen data/evaluation reports, and license notices. It is **not** an already-published GitHub repository. No remote publication or CI execution should be claimed without a real URL and passing GitHub Actions runs.

## Recommended repository

`financial-document-analysis` (descriptive, not branded as a bank project).

## On macOS, after extracting the archive

```bash
brew install gh            # skip if gh is installed
gh auth login              # log into your own GitHub account
cd financial-document-analysis
git init -b main
git add .
git commit -m "Financial research: audited retrieval, sentiment, and arithmetic"
gh repo create bakk0u/financial-document-analysis --public --source=. --remote=origin --push
```

If the repository already exists as an empty GitHub repository, omit `gh repo create` and use `git remote add origin https://github.com/bakk0u/financial-document-analysis.git && git push -u origin main`.

## Verify the published repository

1. Open **Actions** and check actual job results, including the multi-platform Python matrix and Linux Docker smoke; investigate failures instead of claiming success.
2. Check that `.env`, credentials, any `runtime/` uploads, and private documents are excluded by `.gitignore`. Never expose keys.
3. Confirm attribution: FinQA and its MIT license, issuer content provenance, Financial PhraseBank's CC BY-NC-SA 3.0 terms, and the separate noncommercial model/data treatment. Source-code MIT does not override data/model licenses.
4. Review README figures and reports/UNCERTAINTY.md, reports/ARITHMETIC.md, reports/ABLATION.md. Link the public repository from the CV only once it actually resolves and remote CI results are checked.
5. Before a technical interview, retrain one component locally, inspect failure cases, and independently change a test or method you can explain. Do not imply that an AI-assisted one-day build had months of manually authored commit history.

For a private GitHub repository replace `--public` with `--private`. The original dataset permits specified noncommercial use; publication and downstream reuse must comply with its actual license.
