import io
import unittest
from contextlib import redirect_stdout

from tests.setup_checks import check_setup


class SetupCheckTests(unittest.TestCase):
    def test_repository_setup(self):
        with redirect_stdout(io.StringIO()):
            result = check_setup()
        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
