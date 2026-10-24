import unittest
import torch
import numpy as np
from bgeonet.rolling import channel_shift
from bgeonet.raster import RibbonRaster
from bgeonet.geometry import ribbon
from metrics.canonical import case, FIELDS


class CoreTests(unittest.TestCase):
    def test_channel_shift(self):
        x = torch.randn(2, 7, 9, 11, requires_grad=True)
        for axis in (2, 3):
            for sign in (-1, 1):
                reference = torch.cat([torch.roll(c, sign*i, axis) for i, c in enumerate(torch.chunk(x, 7, 1))], 1)
                actual = channel_shift(x, axis, sign)
                self.assertTrue(torch.equal(reference, actual))
                self.assertTrue(torch.equal(torch.autograd.grad(reference.sum(), x, retain_graph=True)[0],
                                            torch.autograd.grad(actual.sum(), x, retain_graph=True)[0]))

    def test_raster_values_and_gradients(self):
        p = torch.tensor([[[.4, .1, .6, .9, .05, .15]]], requires_grad=True)
        expected = ribbon(p)
        actual = RibbonRaster()(p)
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(torch.autograd.grad(expected.sum(), p)[0], torch.autograd.grad(actual.sum(), p)[0]))

    def test_metrics(self):
        self.assertEqual(len(FIELDS), 59)
        gt = np.zeros((512, 512), bool); gt[10:40, 20:30] = True
        perfect = case(gt, gt, 1.)
        self.assertEqual(perfect['dice'], 1.)
        self.assertEqual(perfect['hd95'], 0.)
        empty = case(gt, np.zeros_like(gt), 1.)
        self.assertEqual(empty['dice'], 0.)
        self.assertEqual(empty['hd'], 512.)


if __name__ == '__main__':
    unittest.main()
