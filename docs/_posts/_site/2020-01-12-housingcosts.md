## INTRODUCTION

In this project, the monthly housing cost is predicted by applying several supervised machine learning models available in the Python scikit-learn library. A potential home buyer could benefit from this machine learning application by providing the user a cost estimate of a property before closing the deal. Government agencies could use this application to identify homes in financial distress in a given region, thereby influencing their policy decisions.

## DATA
The data set used is called the Housing Affordability Data System (HADS) which consists of individual datasets spanning the years 1985 to 2013. The data sets are available for download here: [American Housing Survey: Housing Affordability Data System](https://www.huduser.gov/portal/datasets/hads/hads.html). The HADS data are sourced from the American Housing Survey (AHS) national sample microdata and the AHS metropolitan sample microdata. Therefore, more details can be obtained by referring to the AHS data. Each row in the data is an observation of a housing unit. The features fall under 4 categories:

* Cost measures: Utility costs, mortgage payments, HOA fees
* Housing unit characteristics: number of bedrooms, year built, location, structure type
* Household characteristics: income, number of people, rent/owner status
* Local market conditions: median income, fair market rent, poverty level income

To avoid data leakage issues, the features corresponding to cost measures are not fed into the machine learning models. The table below lists the features used to predict the monthly housing cost.
