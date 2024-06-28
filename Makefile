ENV_PATH=~/venv/bin/

dist/pcrscore-0.0.1.tar.gz: src/ pyproject.toml
	$(ENV_PATH)python3 -m build

deploy:
	$(ENV_PATH)python3 -m twine upload --repository testpypi dist/* --skip-existing

install:
	$(ENV_PATH)pip install -i https://test.pypi.org/simple/ pCRscore

local-install:
	$(ENV_PATH)pip install .
