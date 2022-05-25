CS-NER: Computer Science Named Entity Recognition
""""""""""""""""""""""""""""""""""""""""""""""""""

Overview
*********


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
    annotations = annotator.annotate(title='Your paper title here', abstract='Your paper abstract here')
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
    annotations = annotator.annotate_title(title='Your paper title here')
    # or
    annotations = annotator.annotate_abstract(abstract='Your paper abstract here')
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