PYTHON ?= python
STUDY_CONFIG ?= study/study.yaml
STUDY_OUTPUT ?= build/study/canonical

.PHONY: reproduce-study

reproduce-study:
	$(PYTHON) -m gpu_scheduler_lab study run --config $(STUDY_CONFIG)
	$(PYTHON) -m gpu_scheduler_lab study report --input $(STUDY_OUTPUT)
	$(PYTHON) -m gpu_scheduler_lab study verify --input $(STUDY_OUTPUT)
