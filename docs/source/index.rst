.. orkg-nlp documentation master file, created by
   sphinx-quickstart on Fri Apr  8 17:14:03 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to orkg-nlp's documentation!
====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


ORKG-NLP Models
****************

.. list-table::
   :header-rows: 1

   * - ORKG Model
     - Huggingface Repository
     - Description
   * - **cs-ner**
     -
         * `orkg/orkgnlp-cs-ner-titles <https://huggingface.co/orkg/orkgnlp-cs-ner-titles>`_
         * `orkg/orkgnlp-cs-ner-abstracts <https://huggingface.co/orkg/orkgnlp-cs-ner-abstracts>`_

     - Describe me

Environment Variables
**********************

.. list-table::
   :header-rows: 1

   * - Environment Variable
     - Default
     - Description
   * - ORKG_NLP_DATA_CACHE_ROOT
     - ``$USER_HOME/orkgnlp_data``
     - Path of root cache directory where the required models and datasets are downloaded.