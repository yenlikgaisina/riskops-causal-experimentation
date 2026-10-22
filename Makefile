.PHONY: install test demo api
install:
	python -m pip install -r experiment_lab/requirements.txt
test:
	python -m pytest experiment_lab/tests -q
demo:
	python -m experiment_lab.cli --population 20000 --seed 42 --effect 0.05 --train-model
api:
	uvicorn experiment_lab.api:app --reload
