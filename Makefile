PYTHON = python

.PHONY: install baseline rf xgb eda prediction_plots test all clean

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

# EDA notebook
eda:
	$(PYTHON) -m jupyter nbconvert --to notebook --execute EDA_instacart.ipynb --output EDA_instacart_executed.ipynb

# prediction plots notebook
prediction_plots:
	$(PYTHON) -m jupyter nbconvert --to notebook --execute prediction_plot.ipynb --output prediction_plot_executed.ipynb

# Logistic Regression
baseline:
	cd src && $(PYTHON) train_baseline.py

# Random Forest
rf:
	cd src && $(PYTHON) train_rf.py

# XGBoost
xgb:
	cd src && $(PYTHON) train_xgb.py

test:
	cd src && $(PYTHON) test_sanity.py

all: baseline rf xgb test prediction_plots

clean:
	rm -rf __pycache__ src/__pycache__ .pytest_cache
	rm -f EDA_instacart_executed.ipynb prediction_plot_executed.ipynb
