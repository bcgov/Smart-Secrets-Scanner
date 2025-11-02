# Safe Git Commit and Push Script
# Task 49: Reliable git workflow with secret scanning protection

param(
    [string]$Message = "",
    [switch]$NoPush
)

$ErrorActionPreference = "Stop"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Git Commit and Push - Safe Workflow" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Validate git repository
Write-Host "[*] Validating Git Repository..." -ForegroundColor Yellow
try {
    git rev-parse --git-dir | Out-Null
    Write-Host "[OK] Git repository detected`n" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Not a git repository!" -ForegroundColor Red
    exit 1
}

# Check .gitignore
Write-Host "[*] Checking .gitignore..." -ForegroundColor Yellow
if (Test-Path .gitignore) {
    Write-Host "[OK] .gitignore exists`n" -ForegroundColor Green
} else {
    Write-Host "[WARN] .gitignore not found`n" -ForegroundColor Yellow
}

# Check .gitattributes
Write-Host "[*] Checking .gitattributes..." -ForegroundColor Yellow
if (Test-Path .gitattributes) {
    Write-Host "[OK] .gitattributes exists`n" -ForegroundColor Green
} else {
    Write-Host "[WARN] .gitattributes not found`n" -ForegroundColor Yellow
}

# Show status
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Git Status" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan
git status --short
Write-Host ""

# Check for changes
$statusOutput = git status --porcelain
if ([string]::IsNullOrWhiteSpace($statusOutput)) {
    Write-Host "[INFO] No changes to commit. Working tree is clean." -ForegroundColor Cyan
    exit 0
}

# Show diff
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Changes Summary" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan
git diff --stat
Write-Host ""

# Get commit message
if ([string]::IsNullOrWhiteSpace($Message)) {
    Write-Host "Enter commit message (or 'quit' to cancel):" -ForegroundColor Yellow
    $Message = Read-Host "Message"
    if ($Message -eq "quit" -or [string]::IsNullOrWhiteSpace($Message)) {
        Write-Host "`n[CANCELLED] Commit cancelled.`n" -ForegroundColor Yellow
        exit 0
    }
}

# Stage changes
Write-Host "`n[*] Staging all changes..." -ForegroundColor Yellow
git add .
Write-Host "[OK] Changes staged`n" -ForegroundColor Green

# Show staged files
Write-Host "Files staged for commit:" -ForegroundColor Cyan
git diff --cached --name-status
Write-Host ""

# Commit
Write-Host "[*] Creating commit: $Message" -ForegroundColor Yellow
git commit -m $Message
Write-Host "[OK] Commit created`n" -ForegroundColor Green

# Show commit
git log -1 --oneline --decorate
Write-Host ""

# Push
if (-not $NoPush) {
    $currentBranch = git rev-parse --abbrev-ref HEAD
    Write-Host "`n[*] Pushing to remote ($currentBranch)..." -ForegroundColor Yellow
    try {
        git push origin $currentBranch
        Write-Host "`n[SUCCESS] Pushed to remote!`n" -ForegroundColor Green
    } catch {
        Write-Host "`n[ERROR] Push failed!" -ForegroundColor Red
        Write-Host "Your changes are committed locally but NOT pushed." -ForegroundColor Yellow
        Write-Host "To push later: git push origin $currentBranch`n" -ForegroundColor Cyan
        exit 1
    }
} else {
    Write-Host "[INFO] Push skipped (NoPush flag set)" -ForegroundColor Cyan
    $currentBranch = git rev-parse --abbrev-ref HEAD
    Write-Host "To push later: git push origin $currentBranch`n" -ForegroundColor Cyan
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Complete!" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan
