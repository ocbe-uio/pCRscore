PY_PATH=~/venv/bin/python3
PIP_PATH=~/venv/bin/pip

dist/pcrscore-0.0.1.tar.gz: src/ pyproject.toml
	$(PY_PATH) -m build

deploy:
	$(PY_PATH) -m twine upload --repository testpypi dist/* --skip-existing

install:
	$(PIP_PATH) install -i https://test.pypi.org/simple/ pCRscore
