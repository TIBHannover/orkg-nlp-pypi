from unittest import TestCase

from orkgnlp.nli import TemplatesRecommender


class TestTemplatesRecommender(TestCase):

    def setUp(self):
        self.recommender = TemplatesRecommender(force_download=False)
        self.title = 'multimedia ontology learning for automatic annotation and video browsing'
        self.abstract = 'in this work, we offer an approach to combine standard multimedia analysis techniques with knowledge drawn from conceptual metadata provided by domain experts of a specialized scholarly domain, to learn a domain specific multimedia ontology from a set of annotated examples a standard bayesian network learning algorithm that learns structure and parameters of a bayesian network is extended to include media observables in the learning an expert group provides domain knowledge to construct a basic ontology of the domain as well as to annotate a set of training videos these annotations help derive the associations between high level semantic concepts of the domain and low level mpeg 7 based features representing audio visual content of the videos we construct a more robust and refined version of this ontology by learning from this set of conceptually annotated videos to encode this knowledge, we use mowl, a multimedia extension of web ontology language (owl) which is capable of describing domain concepts in terms of their media properties and of capturing the inherent uncertainties involved we use the ontology specified knowledge for recognizing concepts relevant to a video to annotate fresh addition to the video database with relevant concepts in the ontology these conceptual annotations are used to create hyperlinks in the video collection, to provide an effective video browsing interface to the user'
        self.addCleanup(self.recommender.release_memory)

    def test_singleton(self):
        another_recommender = TemplatesRecommender()
        self.assertEqual(self.recommender, another_recommender)

    def test_recommend(self):
        templates = self.recommender(title=self.title, abstract=self.abstract)

        self.assertIsInstance(templates, list)
        for template in templates:
            self.assertIsInstance(template, dict)
            self.assertIn('id', template)
            self.assertIn('label', template)
            self.assertIn('score', template)
