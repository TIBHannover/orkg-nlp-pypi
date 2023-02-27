CS-NER: Computer Science Named Entity Recognition
""""""""""""""""""""""""""""""""""""""""""""""""""

Overview
*********

The ORKG CS-NER system is based on a standardized set of seven contribution-centric scholarly entities viz.,
research problem, solution, resource, language, tool, method, and dataset. It can automatically extract all seven
entity types from Computer Science publication titles. Furthermore, it can extract research problem and method entity
types from Computer Science publication abstracts.

Supported Concepts
^^^^^^^^^^^^^^^^^^
.. list-table::
   :header-rows: 1

   * - Text
     - Concepts
   * - **Title**
     - ``RESEARCH_PROBLEM``, ``SOLUTION``, ``RESOURCE``, ``LANGUAGE``, ``TOOL``, ``METHOD``, ``DATASET``.
   * - **Abstract**
     - ``RESEARCH_PROBLEM``, ``METHOD``.

Usage
******

.. code-block:: python

    from orkgnlp.annotation import CSNer

    annotator = CSNer() # This will also download the required model files.
    annotations = annotator(title='Your paper title here', abstract='Your paper abstract here')
    print(annotations)


and the output has the following schema:

.. code-block:: javascript

    {
        "title": [
            {
                "concept": "some_concept",
                "entities": ["annotated entity", "another annotated entity", ... ]
            }
            ....
        ],
        "abstract": [
            {
                "concept": "some_concept",
                "entities": ["annotated entity", "another annotated entity", ... ]
            }
            ....
        ]
    }

If you don't need to extract the annotations for both the abstract and the title, you can also extract them separately.
E.g:

.. code-block:: python

    from orkgnlp.annotation import CSNer

    annotator = CSNer() # This will also download the required model files.
    annotations = annotator(title='Your paper title here')
    # or
    annotations = annotator(abstract='Your paper abstract here')
    print(annotations)

and then each output has the following schema:

.. code-block:: javascript

    [
        {
            "concept": "some_concept",
            "entities": ["annotated entity", "another annotated entity", ... ]
        }
        ....
    ]
