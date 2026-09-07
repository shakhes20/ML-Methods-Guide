# How to Put This on GitHub — Step by Step

---

## Step 1: Create a GitHub Account (if you don't have one)

1. Go to `https://github.com`
2. Click **Sign up**
3. Choose a username (e.g., `mohammadesky`)
4. Verify your email

---

## Step 2: Create a New Repository

1. Click the **+** button (top right) → **New repository**
2. Fill in:
   ```
   Repository name:  ML-Methods-Guide
   Description:      Methods in Machine Learning — Study Guide (TU Freiberg)
   Visibility:       Public  ← so you can share it
   ```
3. Check: **Add a README file** → NO (we have our own)
4. Click **Create repository**

---

## Step 3: Upload the Files

### Option A: Upload via Browser (Easiest)

1. In your new repo, click **Add file** → **Upload files**
2. Drag and drop all files from the zip
3. At the bottom, write a commit message: `Initial commit — ML study guide`
4. Click **Commit changes**

### Option B: Via Terminal (if you have Git installed)

```bash
# 1. Extract the zip file
unzip ML-Methods-Guide.zip

# 2. Open terminal in that folder
cd ML-Methods-Guide

# 3. Initialize git
git init

# 4. Add all files
git add .

# 5. Commit
git commit -m "Initial commit — ML Methods study guide"

# 6. Connect to GitHub
git remote add origin https://github.com/YOUR_USERNAME/ML-Methods-Guide.git

# 7. Push
git branch -M main
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username.

---

## Step 4: Make It Look Nice (Optional but recommended)

GitHub automatically renders `README.md` as the homepage of your repo.

Your repo will look like:
```
github.com/YOUR_USERNAME/ML-Methods-Guide
```

With a beautiful formatted page showing all your study notes.

---

## Step 5: Share Your Repo

Copy the URL and share:
```
https://github.com/YOUR_USERNAME/ML-Methods-Guide
```

---

## File Structure After Upload

```
ML-Methods-Guide/
│
├── README.md                         ← Homepage of repo
│
├── chapters/
│   ├── 01_supervised_learning.md
│   ├── 02_unsupervised_learning.md
│   ├── 03_generative_models.md
│   └── 04_reinforcement_learning.md
│
├── docs/
│   ├── exam_prep.md
│   └── github_setup.md               ← This file
│
└── assets/
    └── formula_sheet.md
```

---

## Future: How to Update

Whenever you learn something new:

```bash
# Edit a file
# Then:
git add .
git commit -m "Added notes on Q-Learning"
git push
```

Or just edit directly on GitHub website — click any file → pencil icon → edit → commit.

---

## Why GitHub for Study Notes?

```
✅ Free forever
✅ Accessible from any device
✅ Version history — never lose work
✅ Shareable with professors, employers, classmates
✅ Looks professional on your CV
✅ Can add Jupyter notebooks later
✅ Markdown renders beautifully
```
