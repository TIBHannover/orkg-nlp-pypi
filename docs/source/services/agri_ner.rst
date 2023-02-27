Agri-NER: Agriculture Named Entity Recognition
""""""""""""""""""""""""""""""""""""""""""""""""""

Overview
*********

The ORKG Agri-NER system is based on a standardized set of seven contribution-centric scholarly entities viz.,
research problem, solution, resource, language, tool, method, and dataset. It can automatically extract all seven
entity types from Agriculture publication titles.

Supported Concepts
^^^^^^^^^^^^^^^^^^
.. list-table::
   :header-rows: 1

   * - Text
     - Concepts
   * - **Title**
     - ``RESEARCH_PROBLEM``, ``PROCESS``, ``METHOD``, ``RESOURCE``, ``SOLUTION``, ``LOCATION``, ``TECHNOLOGY``.

Usage
******

.. code-block:: python

    from orkgnlp.annotation import AgriNer

    annotator = AgriNer() # This will also download the required model files.
    annotations = annotator(title='Your paper title here')
    print(annotations)


and the output has the following schema:

.. code-block:: javascript

    [
        {
            "concept": "some_concept",
            "entities": ["annotated entity", "another annotated entity", ... ]
        }
        ....
    ]
