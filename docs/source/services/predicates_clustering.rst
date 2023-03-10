Predicates Clustering
""""""""""""""""""""""

Overview
*********

The predicates clustering service implements a recommendation service on the top, based on K-means as a clustering
algorithm. The grouped data points in our clusters are research papers represented by their titles and abstracts.
Data points are semantically grouped based on their research domain contribution and, thus, semantically related
predicates are to be recommended to a specific given research paper. This is beneficial in terms of expediting
structuring a new paper in the ORKG, and of converging towards the usage of shared vocabulary across users.


Usage
******

.. code-block:: python

    from orkgnlp.clustering import PredicatesRecommender

    predicates_recommender = PredicatesRecommender() # This will also download the required model files.
    predicates = predicates_recommender(title='paper title', abstract='long abstract text here')
    print(predicates)

and the output has the following schema:

.. code-block:: javascript

    [
        {
            "id": "some_id",
            "label": "some_label"
        }
        ...
    ]
