from unittest import TestCase

from orkgnlp.common.util.decorators import singleton


@singleton
class A:
    pass


@singleton
class B:
    pass


class C:
    pass


class TestSingleton(TestCase):

    def test_singleton(self):
        a = A()
        another_a = A()

        b = B()
        another_b = B()

        c = C()
        another_c = C()

        self.assertEqual(a, another_a)
        self.assertEqual(b, another_b)
        self.assertNotEqual(c, another_c)
