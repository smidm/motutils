import unittest

from click.testing import CliRunner

from motutils.cli import cli


class CLITestCase(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_visualize_missing_arguments(self):
        result = self.runner.invoke(cli, ["visualize"])
        self.assertEqual(result.exit_code, 2)

        result = self.runner.invoke(cli, ["visualize", "tests/data/Sowbug3_cut.mp4"])
        self.assertEqual(result.exit_code, 2)

        result = self.runner.invoke(
            cli,
            [
                "visualize",
                "tests/data/Sowbug3_cut.mp4",
                "tests/out/cli_visualize_test.mp4",
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 2)

    def test_visualize(self):
        result = self.runner.invoke(
            cli,
            [
                "--load-mot",
                "tests/data/Sowbug3_cut_truncated.csv",
                "visualize",
                "tests/data/Sowbug3_cut.mp4",
                "tests/out/cli_visualize_test.mp4",
                "--limit-duration",
                "1",
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0)

    def test_visualize_multiple(self):
        result = self.runner.invoke(
            cli,
            [
                "--load-mot",
                "tests/data/Sowbug3_cut_truncated.csv",
                "--load-mot",
                "tests/data/Sowbug3_cut_pose.csv",
                "visualize",
                "tests/data/Sowbug3_cut.mp4",
                "tests/out/cli_visualize_test_multiple.mp4",
                "--limit-duration",
                "1",
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0)

    def test_visualize_sleap(self):
        result = self.runner.invoke(
            cli,
            [
                "--load-sleap-analysis",
                "tests/data/sample_sleap.analysis.h5",
                "visualize",
                "tests/data/Sowbug3_cut.mp4",
                "tests/out/cli_visualize_test_sleap.mp4",
                "--limit-duration",
                "1",
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0)

    def test_convert_missing_arguments(self):
        result = self.runner.invoke(cli, ["convert"], catch_exceptions=False)
        self.assertEqual(result.exit_code, 2)
        result = self.runner.invoke(
            cli, ["convert", "tests/out/out_mot.txt"], catch_exceptions=False
        )
        self.assertEqual(result.exit_code, 2)
        result = self.runner.invoke(
            cli,
            [
                "--load-sleap-analysis",
                "tests/data/sample_sleap.analysis.h5",
                "convert",
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 2)
        result = self.runner.invoke(
            cli,
            [
                "--load-sleap-analysis",
                "tests/data/sample_sleap.analysis.h5",
                "--load-mot",
                "tests/data/Sowbug3_cut_pose.csv",
                "convert",
                "tests/out/out_mot.txt",
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 2)

    def test_convert(self):
        result = self.runner.invoke(
            cli,
            [
                "--load-sleap-analysis",
                "tests/data/sample_sleap.analysis.h5",
                "convert",
                "tests/out/sample_sleap.txt",
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0)

    def test_eval_missing_arguments(self):
        result = self.runner.invoke(cli, ["eval"], catch_exceptions=False)
        self.assertEqual(result.exit_code, 2)
        result = self.runner.invoke(
            cli,
            ["eval", "--write-eval", "tests/out/out_mot.txt"],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 2)
        result = self.runner.invoke(
            cli,
            [
                "--load-sleap-analysis",
                "tests/data/sample_sleap.analysis.h5",
                "--load-mot",
                "tests/data/Sowbug3_cut_pose.csv",
                "eval",
                "tests/out/out_mot.txt",
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 2)

    def test_eval(self):
        result = self.runner.invoke(
            cli,
            [
                "--load-toxtrac",
                "tests/data/toxtrac/Tracking_0.txt",
                "--toxtrac-topleft-xy",
                "52",
                "40",
                "--load-gt",
                "tests/data/Sowbug3_cut_truncated.csv",
                "eval",
                "--write-eval",
                "tests/out/out_mot.csv",
            ],
            catch_exceptions=False,
        )
        print(result.stdout)
        self.assertEqual(result.exit_code, 0)
