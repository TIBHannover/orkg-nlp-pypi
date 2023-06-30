Research Fields Classification
"""""""""""""""""""""""""

Overview
*********

This research field classification service aims to predict the corresponding research fields for given papers.
It is designed to assist contributors who may not be familiar with the extensive research field taxonomy present
in the ORKG, enabling them to save significant amounts of time. By analysing the title and abstract of a paper, the service
suggests potential research fields that align with the content. This empowers authors to effortlessly select an
appropriate research field without requiring in-depth knowledge of the research field taxonomy.

Usage
******

.. code-block:: python

    from orkgnlp.annotation import ResearchFieldClassifier

    rf_classifier = ResearchFieldClassifier() # This will also download the required model files.
    rfs = rf_classifier(raw_input='Your paper combined title with abstract here', top_n=10)
    print(rfs)

and the output has the following schema:

.. code-block:: javascript

    [
        {
            "research_field": "some_research_field",
            "score": 0.991233
        }
        ...
    ]
