import unittest
import sys

sys.path.append('/home/kadenw/Workspace/lab/lab')
from lab.loss import (
    cross_entropy,
    entropy,
    kl_divergence,
    rand_probs,
)

class Test_Rand_Probs(unittest.TestCase):

    def test_zero_possbilities(self):
        with self.assertRaises(AssertionError) as context:
            rand_probs(0)
        self.assertEqual(str(context.exception), "n should be above 0")
    
    def test_1_possibility(self):
        self.assertAlmostEqual(sum(rand_probs(1)), 1)

    def test_5_possibilities(self):
        self.assertAlmostEqual(sum(rand_probs(5)), 1)
        
    def test_1000_possibilities(self):
        self.assertAlmostEqual(sum(rand_probs(1000)), 1)

    def test_prob_below_one(self):
        for i in (rand_probs(5)):
            self.assertTrue(i <= 1)

    def test_prob_above_0(self):
        for i in (rand_probs(5)):
            self.assertTrue(i > 0)

class Test_KL_Divergence(unittest.TestCase):

    def test_different_lengths(self):
        with self.assertRaises(ValueError) as context:
            kl_divergence([1],[0.1, 0.9])
        self.assertEqual(str(context.exception), "Distrubitions must have same # of probabilities")

    def test_identical_distrubutions(self):
        self.assertAlmostEqual(kl_divergence([0.4, 0.1, 0.5], [0.4, 0.1, 0.5]), 0) 

    def test_unidentical_distrubutions(self):
        self.assertAlmostEqual(kl_divergence([0.5, 0.5], [0.6, 0.4]), 0.0294468445268)

    def test_prob_with_zero(self):
        self.assertAlmostEqual(kl_divergence([1, 0], [0,1]), 33.21928094555)

class Test_Cross_Entropy(unittest.TestCase):

    def test_different_lengths(self):
        with self.assertRaises(ValueError) as context:
            cross_entropy([1], [0.1, 0.9])
        self.assertEqual(str(context.exception), "Distrubitions must have same # of probabilities")

    def test_prob_with_zero(self):
        self.assertAlmostEqual(cross_entropy([1, 0], [0, 1]), 33.2192809489)

    def test_unidentical_distrubutions(self):
        self.assertAlmostEqual(cross_entropy([0.4, 0.6], [0.2, 0.8]), 1.12192809489)

class Test_Entropy(unittest.TestCase):

    def test_0_probabilities(self):
        self.assertAlmostEqual(entropy([]), 0)
        
    def test_1_probability(self):
        self.assertAlmostEqual(entropy([1]), 0)
    
    def test_known_entropy(self):
        self.assertAlmostEqual(entropy([0.1, 0.6, 0.3]), 1.29546184424)

    def test_uniform_probabilities(self):
        self.assertAlmostEqual(entropy([0.25, 0.25, 0.25, 0.25]), 2) 

if __name__ == '__main__':
    unittest.main()

