TDM-Extraction (Task-Dataset-Metric)
""""""""""""""""""""""""""""""""""""

Overview
*********

Based on our `publication <https://doi.org/10.1007/978-3-030-91669-5_35>`_ this service has been developed as
a Leaderboard mining system from research publications. The service extracts TDM (Task-Dataset-Metric) entities out
of a text represented in DocTAET (Title, Abstract, ExperimentalSetup and TableInformation) representation.

We provide a DocTAET parser from PDF files in
`this repository <https://github.com/Kabongosalomon/task-dataset-metric-extraction>`_ and you can also find our
gold TDM labels on `huggingface <https://huggingface.co/orkg/orkgnlp-tdm-extraction/blob/main/labels.tsv>`_.

Usage
******

.. code-block:: python

    from orkgnlp.annotation import TdmExtractor

    tdm_extractor = TdmExtractor() # This will also download the required model files.
    tdms = tdm_extractor(text='DocTAET represented text here', top_n=10)
    print(tdms)

and the output has the following schema:

.. code-block:: javascript

    [
        {
            "task": "some_task",
            "dataset": "some_dataset",
            "metric": "some_metric",
            "score": 0.991233
        }
        ...
    ]
