# Git Workflow Quick Reference

## Using the Safe Git Commit Script

### Basic Usage (Interactive Mode)
```powershell
.\git-commit-push.ps1
```
- Shows status and diff
- Prompts for commit message
- Commits and pushes to remote

### With Commit Message
```powershell
.\git-commit-push.ps1 -Message "Add new feature"
```
- Skips commit message prompt
- Commits and pushes immediately

### Commit Only (No Push)
```powershell
.\git-commit-push.ps1 -Message "Work in progress" -NoPush
```
- Commits locally
- Does NOT push to remote

---

## What the Script Does

1. ✅ **Validates** git repository and configuration files
2. ✅ **Shows** current status and changes
3. ✅ **Prompts** for review and confirmation
4. ✅ **Stages** all changes with `git add .`
5. ✅ **Commits** with your message
6. ✅ **Pushes** to remote (unless -NoPush)
7. ✅ **Handles errors** with rollback option

---

## Protected Files (Won't Cause Secret Scanning Issues)

### Already Excluded (in .gitignore):
- `scripts/generate_simple_training_data.py` - Contains example secret patterns
- `data/processed/**/*.jsonl` - Training data files
- `models/**/*.bin` and `*.safetensors` - Large model files
- `outputs/logs/*.tfevents.*` - TensorBoard logs
- `offload/` - Model offload directory
- `models/base/.locks/` - HuggingFace cache locks
- Environment files: `.env`, `*.key`, `*.pem`

### Marked as Safe Documentation (in .gitattributes):
- `research/**/*.md` - Research documents with example secrets
- `tasks/**/*.md` - Task documentation
- `*.md` - All markdown files (educational content)

---

## If Git Push Fails Due to Secrets

### Option 1: Files are Already Protected
The script should handle this automatically with `.gitattributes`

### Option 2: Add Specific File to .gitignore
```powershell
# Edit .gitignore and add the problematic file
code .gitignore
```

### Option 3: Use Commit Without Push
```powershell
# Commit locally first, review GitHub feedback
.\git-commit-push.ps1 -NoPush

# Review what GitHub flags
# Fix if needed
# Then push manually
git push
```

---

## Manual Git Commands (If Needed)

### Check Status
```powershell
git status
```

### View Changes
```powershell
git diff
git diff --stat
```

### Stage Specific Files Only
```powershell
git add README.md
git add scripts/fine_tune.py
```

### Commit Staged Changes
```powershell
git commit -m "Your message here"
```

### Push to Remote
```powershell
git push
```

### Undo Last Commit (Keep Changes)
```powershell
git reset --soft HEAD~1
```

### Unstage All Files
```powershell
git reset
```

---

## Checking Protection is Working

### Verify .gitignore is Active
```powershell
# This should NOT show generate_simple_training_data.py
git status
```

### Verify .gitattributes is Active
```powershell
# Check if markdown files are marked as documentation
git check-attr linguist-documentation research/edge_case_improvements.md
# Output should be: linguist-documentation: true
```

### Test Safe Commit
```powershell
# Try a test commit (without push)
.\git-commit-push.ps1 -Message "Test commit" -NoPush
# Should succeed without warnings

# If successful, undo it
git reset --soft HEAD~1
```

---

## Common Issues and Solutions

### Issue: "Not a git repository"
**Solution:** Run from project root directory
```powershell
cd C:\Users\RICHFREM\source\repos\Llama3-FineTune-Coding
```

### Issue: "Permission denied" or authentication failure
**Solution:** Check GitHub authentication
```powershell
# Check if you have a GitHub token configured
git config --global credential.helper

# Or use SSH authentication
git remote set-url origin git@github.com:bcgov/Smart-Secrets-Scanner.git
```

### Issue: Secret scanning blocks push
**Solution:** Verify file is in .gitignore or .gitattributes
```powershell
# Check .gitignore
cat .gitignore | Select-String "problematic-file.py"

# Check .gitattributes
cat .gitattributes | Select-String "linguist-documentation"
```

### Issue: Large files rejected
**Solution:** Files should be in .gitignore
```powershell
# Model files are already excluded
# If needed, add to .gitignore:
# models/**/*.bin
# models/**/*.safetensors
```

---

## Best Practices

1. **Always review changes** before committing
   - Use the script's built-in diff review
   - Or run `git diff` manually

2. **Write clear commit messages**
   - Use present tense: "Add feature" not "Added feature"
   - Be descriptive: "Add Task 49 - git workflow script" not "Update"

3. **Commit related changes together**
   - Don't mix unrelated features in one commit
   - Make small, focused commits

4. **Use -NoPush for work-in-progress**
   - Commit locally while working
   - Push when ready to share

5. **Keep sensitive data out of repo**
   - Never commit real API keys, passwords, tokens
   - Use environment variables or secret managers
   - Example secrets in docs are OK (marked safe)

---

## Quick Commands Cheat Sheet

| Task | Command |
|------|---------|
| Safe commit & push | `.\git-commit-push.ps1` |
| Commit only (no push) | `.\git-commit-push.ps1 -NoPush` |
| With message | `.\git-commit-push.ps1 -Message "msg"` |
| Check status | `git status` |
| View changes | `git diff` |
| Undo last commit | `git reset --soft HEAD~1` |
| Unstage all | `git reset` |
| Manual push | `git push` |
| Check gitignore works | `git status` (excluded files won't show) |
| Check gitattributes | `git check-attr linguist-documentation file.md` |

---

## Related Documentation

- **Task 49:** `tasks/done/49-create-git-commit-push-script.md`
- **Git Script:** `git-commit-push.ps1`
- **Git Ignore:** `.gitignore`
- **Git Attributes:** `.gitattributes`

---

**Last Updated:** 2025-11-02  
**Version:** 1.0
