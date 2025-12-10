PYTHON = python

.PHONY: install baseline rf xgb eda all clean

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

# EDA notebook
eda:
	$(PYTHON) -m jupyter nbconvert --to notebook --execute EDA_instacart.ipynb --output EDA_instacart_executed.ipynb

# Logistic Regression
baseline:
	cd src && $(PYTHON) train_baseline.py

# Random Forest
rf:
	cd src && $(PYTHON) train_rf.py

# XGBoost
xgb:
	cd src && $(PYTHON) train_xgb.py

all: baseline rf xgb

clean:
	rm -rf __pycache__ src/__pycache__ .pytest_cache
	rm -f EDA_instacart_executed.ipynb
