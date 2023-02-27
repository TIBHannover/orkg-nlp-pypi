BioAssays Semantification
"""""""""""""""""""""""""

Overview
*********

The bioassay semantification service automatically semantifies bioassay descriptions based on the semantic model
of the `Bioassay ontology <http://bioassayontology.org/>`_. More information on the supporting clustering algorithm of
the service implementation, its development gold-standard dataset, and its performance results can be found
in our `publication <https://doi.org/10.48550/arXiv.2111.15182>`_.

Usage
******

.. code-block:: python

    from orkgnlp.clustering import BioassaysSemantifier

    bioassays_semantifier = BioassaysSemantifier() # This will also download the required model files.
    labels = bioassays_semantifier(text='BioAssay text description here')
    print(labels)


and the output has the following schema:

.. code-block:: javascript

    [
        {
            "property": {
                "id": "some_id",
                "label": "some_label"
            },
            "resources": [
                {
                    "id": "some_id",
                    "label": "some_label"
                }
                ...
            ]
        }
        ...
    ]
