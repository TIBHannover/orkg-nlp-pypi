from unittest import TestCase

from orkgnlp.clustering import PredicatesRecommender


class TestPredicatesRecommender(TestCase):

    def setUp(self):
        self.recommender = PredicatesRecommender()

    def test_singleton(self):
        another_recommender = PredicatesRecommender()
        self.assertEqual(self.recommender, another_recommender)

    def test_recommend(self):
        title = 'Knowledge modelling in weakly‐structured business processes '
        abstract = 'in paper present new approach integrating knowledge management business process management we focus modelling weakly\u2010structured knowledge\u2010intensive business process we develop framework modelling type process explicitly considers knowledge\u2010related task knowledge object present workflow tool implementation theoretical meta\u2010model a example sketch one case study process granting full old age pension performed greek social security institution finally briefly describe related approach compare work draw main conclusion research direction'
        self.assertIsNotNone(self.recommender.recommend(title=title, abstract=abstract))