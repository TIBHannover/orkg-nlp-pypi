Basic Usage
============
In order to use ``orkgnlp`` you need to have a look at the :doc:`services/services` we offer. Each service requires a set of files
representing datasets, models and/or configurations. But do not worry, you are not asked to download them manually!

The ``orkgnlp.downloader`` module is a tool for downloading the required files per service. For instance, if you want to use
the ``predicates-clustering`` service, you basically need to download its dependencies by calling

.. code-block:: python

    import orkgnlp
    orkgnlp.download('predicates-clustering')


and you can also download dependencies for multiple services by passing a list of service names like

.. code-block:: python

    import orkgnlp
    orkgnlp.download(['service name', 'another service name'])



If you are a fan of reducing lines of code, you can use the service right away! It will downloads the required dependencies,
if you have not yet! Check this example.


.. code-block:: python

    from orkgnlp.clustering import PredicatesRecommender

    predicates_recommender = PredicatesRecommender() # This will also download the required model files.
    predicates = predicates_recommender(title='paper title', abstract='long abstract text here')
    print(predicates)

    # output: [{"id": "P1234", "label": "some predicate"}, {"id": "P4321", "label": "another predicate"}]

.. note::
    Once a service's dependencies are downloaded they will be cached and not downloaded again as long as they have not
    changed. Currently we store our model and data files on ``huggingface`` and use the caching concept of its python
    client.

.. note::
    The files will be downloaded to the default root directory ``$USER_HOME/orkgnlp_data``. Check :doc:`configure`
    in case you want to change it.