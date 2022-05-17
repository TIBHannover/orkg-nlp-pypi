BioAssays Semantification
"""""""""""""""""""""""""

Overview
*********

TODO

Usage
******

.. code-block:: python

    from orkgnlp.clustering import BioassaysSemantifier

    bioassays_semantifier = BioassaysSemantifier() # This will also download the required model files.
    result = bioassays_semantifier.semantify(text='BioAssay text description here')
    print(result)


and the output has the following schema:

.. code-block:: javascript

    {
        "properties": [
            {
                "id": "some_id",
                "label": "some_label"
            }
            ...
        ],
        "resources": [
            {
                "id": "some_id",
                "label": "some_label"
            }
            ...
        ],
        "labels": {
            "some_property_label": "some_resource_label",
            "another_property_label": [
                "some_resource_label",
                "another_resource_label"
                ...
            ],
            ...
        }
    }
