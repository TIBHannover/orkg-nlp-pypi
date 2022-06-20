from unittest import TestCase

from orkgnlp.clustering import PredicatesRecommender


class TestPredicatesRecommender(TestCase):

    def setUp(self):
        self.recommender = PredicatesRecommender(force_download=False)
        self.addCleanup(self.recommender._release_memory)

    def test_singleton(self):
        another_recommender = PredicatesRecommender()
        self.assertEqual(self.recommender, another_recommender)

    def test_recommend(self):
        title = 'long title'
        abstract = 'long abstract'
        self.assertIsInstance(self.recommender(title=title, abstract=abstract), list)
