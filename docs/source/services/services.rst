ORKG-NLP Services
==================

Supported Services
""""""""""""""""""""""
.. list-table::
   :header-rows: 1

   * - ORKG Service
     - Huggingface Repository
     - Description
   * - ``predicates-clustering``
     - `orkg/orkgnlp-predicates-clustering <https://huggingface.co/orkg/orkgnlp-predicates-clustering>`_
     - Recommendation service for ORKG predicates based on clustering.
   * - ``bioassays-semantification``
     - `orkg/orkgnlp-bioassays-semantification <https://huggingface.co/orkg/orkgnlp-bioassays-semantification>`_
     - Semantification service for BioAssays based on clustering.
   * - ``cs-ner``
     -
        * `orkg/orkgnlp-cs-ner-titles <https://huggingface.co/orkg/orkgnlp-cs-ner-titles>`_
        * `orkg/orkgnlp-cs-ner-abstracts <https://huggingface.co/orkg/orkgnlp-cs-ner-abstracts>`_
     - Annotation service for research papers based on named entity recognition.
   * - ``tdm-extraction``
     - `orkg/orkgnlp-tdm-extraction <https://huggingface.co/orkg/orkgnlp-tdm-extraction>`_
     - Annotation service for Task-Dataset-Metric (TDM) extraction of research papers.

.. include:: ./predicates_clustering.rst
.. include:: ./bioassays_semantification.rst
.. include:: ./cs_ner.rst
.. include:: ./tdm_extraction.rst