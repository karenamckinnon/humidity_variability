====================
humidity_variability
====================

This repo contains code to fit the noncrossing smoothing splines quantile regression model in McKinnon and Poppick, under review in JABES.

Code is research (not production) quality, and will likely need modifications to run on your machine. 

* ./humidity_variability/scripts/save_gsod.py: Get GSOD data used in analysis
* ./humidity_variability/scripts/create_cases.py: Create synthetic data and fit model to the synthetic case studies
* ./humidity_variability/main.py: Fit either the fully interaction model or a simpler linear model to GSOD data. This function is designed to run across multiple processors if available.

Note that the interaction model takes 2-6 seconds to run for each quantile, so fitting a set of 19 quantiles (5th to 95th percentile in steps of 5%) will take about a minute.

Please contact Karen McKinnon (kmckinnon@ucla.edu) if you are using the code or methods in your work.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
