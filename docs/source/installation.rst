Installation
============
Currently we support the installation with :ref:`pip <with pip>` and :ref:`manually <manually>`

.. _with pip:

With PIP
"""""""""

You can simply install the package from `PyPI <https://pypi.org/project/orkgnlp/>`_ by executing the following command:

.. code-block:: bash

    pip install orkgnlp

.. _manually:

Manually
"""""""""
You can also install the package manually by cloning the repository and building the package using `Poetry <https://python-poetry.org/>`_ and then installing it with pip.

.. note::
    You need to replace ``x.x.x`` in the following command with the latest version mentioned in ``orkg-nlp-pypi/pyproject.toml``
.. code-block:: bash

    git clone https://gitlab.com/TIBHannover/orkg/nlp/orkg-nlp-pypi.git
    cd orkg-nlp-pypi
    poetry build
    pip install dist/orkgnlp-x.x.x-py3-none-any.whl # consider replacing x.x.x with the latest version

Verify Installation
"""""""""""""""""""
Now you can verify your installation with the following line:

.. code-block:: python

    import orkgnlp


If no error pops up, you are free to enjoy it! Check our :doc:`basic usage <usage>` or jump right away
to our :doc:`services <services/services>`.