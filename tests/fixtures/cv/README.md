# Synthetic CVs

Everything in this folder is **invented**. "Alex Example" is not a real person, the employers do
not exist, and the addresses, phone numbers and links use reserved or fictional values
(`example.com`, `555` numbers).

These files feed tests, docs and screenshots. A real CV never goes in this repository: not here,
not anywhere else. A pre-commit hook and a test (`scripts/check_no_cv_files.py`) reject any
PDF, and any file named like a CV, outside this folder.

Backups of the database (`pg_dump`) and run records written under `.local/` can contain CV text.
Keep them outside the repository.
