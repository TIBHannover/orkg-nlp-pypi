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
     - `v0.2.0 <https://gitlab.com/TIBHannover/orkg/nlp/experiments/orkg-predicates-clustering/-/releases/v0.2.0>`_
     - `orkg/orkgnlp-predicates-clustering <https://huggingface.co/orkg/orkgnlp-predicates-clustering>`_
     - Recommendation service for ORKG predicates based on clustering.
   * - ``bioassays-semantification``
     - `v0.1.0 <https://gitlab.com/TIBHannover/orkg/nlp/experiments/orkg-bioassays-semantification/-/releases/v0.1.0>`_
     - `orkg/orkgnlp-bioassays-semantification <https://huggingface.co/orkg/orkgnlp-bioassays-semantification>`_
     - Semantification service for BioAssays based on clustering.
   * - ``cs-ner``
     - `v0.1.0 <https://gitlab.com/TIBHannover/orkg/nlp/experiments/orkg-cs-ner/-/releases/v0.1.0>`_
     -
        * `orkg/orkgnlp-cs-ner-titles <https://huggingface.co/orkg/orkgnlp-cs-ner-titles>`_
        * `orkg/orkgnlp-cs-ner-abstracts <https://huggingface.co/orkg/orkgnlp-cs-ner-abstracts>`_
     - Annotation service for research papers in the Computer Science domain based on named entity recognition.
   * - ``tdm-extraction``
     - `v0.1.0`
     - `orkg/orkgnlp-tdm-extraction <https://huggingface.co/orkg/orkgnlp-tdm-extraction>`_
     - Annotation service for Task-Dataset-Metric (TDM) extraction of research papers.
   * - ``templates-recommendation``
     - `v0.1.0 <https://gitlab.com/TIBHannover/orkg/nlp/experiments/orkg-templates-recommendation/-/releases/v0.1.0>`_
     - `orkg/orkgnlp-templates-recommendation <https://huggingface.co/orkg/orkgnlp-templates-recommendation>`_
     - Recommendation service for ORKG templates based on Natural Language Inference (NLI).
   * - ``agri-ner``
     - `v0.1.0 <https://gitlab.com/TIBHannover/orkg/nlp/experiments/orkg-agriculture-ner/-/releases/v0.1.0>`_
     - `orkg/orkgnlp-agri-ner <https://huggingface.co/orkg/orkgnlp-agri-ner>`_
     - Annotation service for research papers in the Agriculture domain based on named entity recognition.
   * - ``research-fields-classification``
     - `v0.1.0`
     - `orkg/orkgnlp-research-fields-classification <https://huggingface.co/orkg/orkgnlp-research-fields-classification>`_
     - Classification service for research field identifications in different domains based on multi-class classification.
(*) Please refer to the release notes or README.md file in the release assets for more information about the version.

To get started with any ORKG NLP service, you can use ``orkgnlp.load()`` and pass the service name from the table above.

.. code-block:: python

    import orkgnlp
    service = orkgnlp.load('predicates-clustering') # This will also download the required model files.
    predicates = service(title='paper title', abstract='long abstract text here')

    service = orkgnlp.load('tdm-extraction') # This will also download the required model files.
    tdms = service(text='DocTAET represented text here', top_n=10)

Read more about each service below!

.. include:: ./predicates_clustering.rst
.. include:: ./bioassays_semantification.rst
.. include:: ./cs_ner.rst
.. include:: ./tdm_extraction.rst
.. include:: ./templates_recommendation.rst
.. include:: ./agri_ner.rst
.. include:: ./research_fields_classification.rst
