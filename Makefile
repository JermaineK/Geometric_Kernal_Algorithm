PYTHON ?= python

.PHONY: test stress stress-nightly

test:
	$(PYTHON) -m pytest -q

stress:
	$(PYTHON) tests/stress/run_stress_suite.py \
		--configs tests/stress/configs/*.yaml \
		--runs 6 \
		--outroot tests/stress/outputs/ci \
		--seed 20260213 \
		--robustness-samples 24 \
		--blind-n 40 \
		--no-cli

stress-nightly:
	$(PYTHON) tests/stress/run_stress_suite.py \
		--configs tests/stress/configs/*.yaml \
		--runs 30 \
		--outroot tests/stress/outputs/nightly \
		--seed 20260213 \
		--robustness-samples 120 \
		--blind-n 100 \
		--no-cli
