# Owner(s): ["oncall: jit"]

import torch
import os
import sys
from torch.testing._internal.jit_utils import JitTestCase

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

class TestModules(JitTestCase):
    def test_script_module_with_constants_list(self):
        """
        Test that a module that has __constants__ set to something
        that is not a set can be scripted.
        """

        # torch.nn.Linear has a __constants__ attribute defined
        # and intialized to a list.
        class Net(torch.nn.Linear):
            x: torch.jit.Final[int]

            def __init__(self):
                super().__init__(5, 10)
                self.x = 0

        self.checkModule(Net(), (torch.randn(5),))

    def test_script_inherited_modules_with_annotations(self):
        """
        Even though inheritence isn't officially supported... people have used it
        in cases where it didn't error out in the past. 3.10 had a regression
        due to how inspect.get_annotation was being used, this checks the situation
        that had regressed.
        """

        class Parent(torch.nn.Module):
            x: List[int]

            def __init__(self):
                super().__init__()
                self.x = [1, 2]

        class Child(Parent):
            y: List[int]

            def __init__(self):
                super().__init__()
                self.y = [3, 4]

            def forward(self, val):
                for v, w in zip(self.x, self.y):
                    val += v * w

                return val

        mod = Child()
        self.checkModule(mod, (torch.rand(4, 4)))
