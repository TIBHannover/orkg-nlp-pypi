Configuration
==============

Configurable Variables
"""""""""""""""""""""""
.. list-table::
   :header-rows: 1

   * - Variable
     - Default
     - Function
     - Description
   * - ORKG_NLP_DATA_CACHE_ROOT
     - ``$USER_HOME/orkgnlp_data``
     - ``orkgnlp.config.set_data_cache_root``
     - Path of root cache directory where the required models and datasets are downloaded.

How to Configure
"""""""""""""""""
We provide an interface to change some global package configurations such as the root cache directory of
the ``downloader`` or the logger verbosity.

In order to change a specific variable you need to call the corresponding setter method from the ``orkgnlp.config``
module.

Example:

.. code-block:: python

    import orkgnlp
    orkgnlp.config.set_data_cache_root('/your/root/path')
    orkgnlp.download('service-name') # the downloader will use your custom-defined path!
