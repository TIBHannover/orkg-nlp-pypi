Templates Recommendation
""""""""""""""""""""""""

Overview
*********

This service aims to foster constructing the ORKG using predefined set of predicates that
are represented by semantic building blocks called Templates. This directs ORKG
users to converge towards selecting predicates added by domain experts while not preventing
them from adding new ones / selecting other ones, as the crowdsourcing concept of the
ORKG suggests. The recommender is based on fine-tuning the `SciBERT <https://aclanthology.org/D19-1371/>`_ pre-trained
model with a linear layer on the top to solve the task as a Natural Language Inference (NLI) problem.
Note that this service and the ``Predicates Clustering`` serve the same purpose, but
from different perspectives. You can find our
gold templates on `huggingface <https://huggingface.co/orkg/orkgnlp-templates-recommendation/blob/main/labels.json>`_.


Usage
******

.. code-block:: python

    from orkgnlp.nli import TemplatesRecommender

    templates_recommender = TemplatesRecommender() # This will also download the required model files.
    templates = templates_recommender(title='paper title', abstract='long abstract text here', top_n=10)
    print(templates)

and the output has the following schema:

.. code-block:: javascript

    [
        {
            "id": "some_id",
            "label": "some_label",
            "score": 0.991233
        }
        ...
    ]
