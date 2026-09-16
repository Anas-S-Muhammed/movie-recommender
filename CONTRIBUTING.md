# Contributing

Contributions are welcome through focused pull requests.

1. Fork the repository and create a branch from `main`.
2. Create a virtual environment and install `requirements-dev.txt`.
3. Keep recommendation logic inside `src/movie_recommender/` and UI logic inside `app.py`.
4. Add or update tests for every behaviour change.
5. Run `ruff check .` and `pytest` before opening a pull request.

For substantial algorithm changes, include the evaluation command and resulting ranking metrics in the pull request.
