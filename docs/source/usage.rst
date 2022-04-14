Basic Usage
============
For the list of supported NLP services please refer to :doc:`services`.

.. autofunction:: orkgnlp.download

.. code-block:: python

    import orkgnlp
    from orkgnlp.annotators import cs_annotator
    orkgnlp.download('cs-ner')

    annotations = cs_annotator.annotate('long text')
    print(annotations)