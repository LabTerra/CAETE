# Documentação CAETE

**Este repositório possui um **[Github Actions](.github/workflows/deploy-sphinx.yml)** configurado para fazer deploy automático da documentação utilizando sphinx.

## Build

```bash
python -m venv ~/py_envs/sphinx
source ~/py_envs/sphinx/bin/activate
pip install -r docs/requirements.txt
sphinx-apidoc -o docs/modules src/ --force
sphinx-apidoc -o docs/modules src/ --force --module-first --separate
sphinx-build -b html docs/ docs/_build
```