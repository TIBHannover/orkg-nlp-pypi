ORKG-NLP Services
==================

Supported Services
""""""""""""""""""""""
.. list-table::
   :header-rows: 1

   * - ORKG Service
     - Version*
     - Huggingface Repository
     - Description
   * - ``predicates-clustering``
     - `v0.1.0 <https://gitlab.com/TIBHannover/orkg/nlp/experiments/orkg-predicates-clustering/-/releases/v0.1.0>`_
     - `orkg/orkgnlp-predicates-clustering <https://huggingface.co/orkg/orkgnlp-predicates-clustering>`_
     - Recommendation service for ORKG predicates based on clustering.
   * - ``bioassays-semantification``
     - `v0.1.0`
     - `orkg/orkgnlp-bioassays-semantification <https://huggingface.co/orkg/orkgnlp-bioassays-semantification>`_
     - Semantification service for BioAssays based on clustering.
   * - ``cs-ner``
     - `v0.1.0`
     -
        * `orkg/orkgnlp-cs-ner-titles <https://huggingface.co/orkg/orkgnlp-cs-ner-titles>`_
        * `orkg/orkgnlp-cs-ner-abstracts <https://huggingface.co/orkg/orkgnlp-cs-ner-abstracts>`_
     - Annotation service for research papers based on named entity recognition.
   * - ``tdm-extraction``
     - `v0.1.0`
     - `orkg/orkgnlp-tdm-extraction <https://huggingface.co/orkg/orkgnlp-tdm-extraction>`_
     - Annotation service for Task-Dataset-Metric (TDM) extraction of research papers.
   * - ``templates-recommendation``
     - `v0.1.0 <https://gitlab.com/TIBHannover/orkg/nlp/experiments/orkg-templates-recommendation/-/releases/v0.1.0>`_
     - `orkg/orkgnlp-templates-recommendation <https://huggingface.co/orkg/orkgnlp-templates-recommendation>`_
     - Recommendation service for ORKG templates based on Natural Language Inference (NLI).

(*) Please refer to the release notes or README.md file in the release assets for more information about the version.

.. include:: ./predicates_clustering.rst
.. include:: ./bioassays_semantification.rst
.. include:: ./cs_ner.rst
.. include:: ./tdm_extraction.rst
.. include:: ./templates_recommendation.rst