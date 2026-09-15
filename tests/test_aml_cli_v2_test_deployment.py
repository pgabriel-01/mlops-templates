import os
import stat
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = REPOSITORY_ROOT / "templates/aml-cli-v2/test-deployment.yml"


def extract_inline_script() -> str:
    lines = TEMPLATE_PATH.read_text(encoding="utf-8").splitlines()
    marker_index = next(
        index for index, line in enumerate(lines) if line.strip() == "inlineScript: |"
    )
    body_lines = lines[marker_index + 1 :]
    return textwrap.dedent("\n".join(body_lines[: body_lines.index("    env:")]))


class TestDeploymentTemplateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temporary_directory.name)
        self.call_log = self.temp_path / "az-calls.log"
        self.mock_bin = self.temp_path / "bin"
        self.mock_bin.mkdir()
        mock_az = self.mock_bin / "az"
        mock_az.write_text(
            textwrap.dedent(
                """\
                #!/usr/bin/env bash
                set -euo pipefail
                [[ "${AZURE_STORAGE_AUTH_MODE:-}" == "login" ]]
                printf '%s\\n' "$*" >> "$AZ_CALL_LOG"

                case "$*" in
                  "ml online-endpoint invoke "*)
                    printf '{"result":"ok"}\\n'
                    ;;
                  "ml batch-endpoint invoke "*)
                    printf 'parent-job\\n'
                    ;;
                  "ml job stream "*)
                    if [[ "$AZ_SCENARIO" == "batch-failure" ]]; then
                      exit 42
                    fi
                    ;;
                  "ml job show "*)
                    if [[ "$AZ_SCENARIO" == "batch-failure" ]]; then
                      printf 'Failed\\n'
                    else
                      printf 'Completed\\n'
                    fi
                    ;;
                  "ml job list "*)
                    printf 'failed-child-run\\n'
                    ;;
                  "ml job download "*)
                    download_path=""
                    while [[ "$#" -gt 0 ]]; do
                      if [[ "$1" == "--download-path" ]]; then
                        download_path="$2"
                        break
                      fi
                      shift
                    done
                    mkdir -p "$download_path/failed-child-run/user_logs"
                    cat > "$download_path/failed-child-run/user_logs/std_log_0.txt" <<'EOF'
                User process exited with code 42.
                https://storage.invalid/container?sig=live-secret
                Authorization: Bearer live-token
                client_secret=live-client-secret
                EOF
                    ;;
                  *)
                    printf 'Unexpected az arguments: %s\\n' "$*" >&2
                    exit 99
                    ;;
                esac
                """
            ),
            encoding="utf-8",
        )
        mock_az.chmod(mock_az.stat().st_mode | stat.S_IXUSR)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def run_script(self, scenario: str, endpoint_type: str) -> subprocess.CompletedProcess:
        environment = os.environ.copy()
        environment.update(
            {
                "AZ_CALL_LOG": str(self.call_log),
                "AZ_SCENARIO": scenario,
                "DEPLOYMENT_NAME": "deployment",
                "ENDPOINT_NAME": "endpoint",
                "ENDPOINT_TYPE": endpoint_type,
                "PATH": f"{self.mock_bin}{os.pathsep}{environment['PATH']}",
                "REQUEST_TYPE": "uri_file",
                "RESOURCE_GROUP": "resource-group",
                "SAMPLE_REQUEST": "azureml:data@latest",
                "WORKSPACE_NAME": "workspace",
            }
        )
        return subprocess.run(
            ["bash", "-c", extract_inline_script()],
            check=False,
            capture_output=True,
            cwd=REPOSITORY_ROOT,
            env=environment,
            text=True,
        )

    def test_online_invocation_uses_keyless_storage_auth(self) -> None:
        result = self.run_script("online-success", "online")

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("ml online-endpoint invoke", self.call_log.read_text())

    def test_successful_batch_job_streams_and_completes(self) -> None:
        result = self.run_script("batch-success", "batch")

        self.assertEqual(0, result.returncode, result.stderr)
        calls = self.call_log.read_text()
        self.assertIn("ml batch-endpoint invoke", calls)
        self.assertIn("ml job stream", calls)
        self.assertIn("ml job show", calls)
        self.assertNotIn("ml job download", calls)

    def test_failed_batch_job_downloads_and_prints_redacted_child_log(self) -> None:
        result = self.run_script("batch-failure", "batch")

        self.assertEqual(1, result.returncode)
        calls = self.call_log.read_text()
        self.assertIn("ml job list --parent-job-name parent-job", calls)
        self.assertIn("ml job download --name failed-child-run", calls)
        self.assertIn("User process exited with code 42.", result.stdout)
        self.assertIn("sig=[REDACTED]", result.stdout)
        self.assertIn("Authorization: Bearer [REDACTED]", result.stdout)
        self.assertIn("client_secret=[REDACTED]", result.stdout)
        self.assertNotIn("live-secret", result.stdout)
        self.assertNotIn("live-token", result.stdout)


if __name__ == "__main__":
    unittest.main()
