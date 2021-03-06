# spb-cap

In this project, the monthly housing cost is predicted by applying several supervised machine learning models available in the Python scikit-learn library. A potential home buyer could benefit from this machine learning application by providing the user a cost estimate of a property before closing the deal. Government agencies could use this application to identify homes in financial distress in a given region, thereby influencing their policy decisions.

The data set used is called the Housing Affordability Data System (HADS) which consists of individual datasets spanning the years 1985 to 2013. The data sets are available for download here: [American Housing Survey: Housing Affordability Data System](https://www.huduser.gov/portal/datasets/hads/hads.html).

Relevant notebooks:

* linear_rf_models.ipynb - use linear ridge and random forest models to predict monthly housing costs

* rf_param_tuning.ipynb - random forest parameter tuning using RandomizedSearchCV instance

* rf_param_tuning2.ipynb - random forest parameter tuning using GridSearchCV instance

* svm.ipynb - support vector machine model used to predict binary label

* data_clean_up_other_years.ipynb - clean up original HADS data files
