import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = REPOSITORY_ROOT / "templates/infra/terraform-deploy.yml"


class TerraformDeployTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.template = TEMPLATE_PATH.read_text(encoding="utf-8")

    def test_platform_network_parameters_are_optional_strings(self) -> None:
        for parameter in (
            "platformResourceGroupName",
            "platformVirtualNetworkName",
        ):
            contract = (
                f"- name: {parameter}\n"
                "    type: string\n"
                "    default: ''"
            )
            self.assertIn(contract, self.template)

    def test_platform_network_parameters_are_passed_to_terraform(self) -> None:
        expected_contracts = (
            (
                "platformResourceGroupName",
                "TF_VAR_PLATFORM_RESOURCE_GROUP_NAME",
                "platform_resource_group_name",
            ),
            (
                "platformVirtualNetworkName",
                "TF_VAR_PLATFORM_VIRTUAL_NETWORK_NAME",
                "platform_virtual_network_name",
            ),
        )

        for parameter, environment_variable, terraform_variable in expected_contracts:
            self.assertIn(
                f"{environment_variable}: ${{{{ parameters.{parameter} }}}}",
                self.template,
            )
            self.assertIn(
                f'-var "{terraform_variable}=${environment_variable}"',
                self.template,
            )

    def test_existing_platform_connectivity_import_is_normalized_and_passed(self) -> None:
        self.assertIn(
            "- name: importExistingPlatformConnectivity\n"
            "    type: boolean\n"
            "    default: false",
            self.template,
        )
        self.assertIn(
            'TF_VAR_IMPORT_EXISTING_PLATFORM_CONNECTIVITY="$(normalize_bool '
            '"$TF_VAR_IMPORT_EXISTING_PLATFORM_CONNECTIVITY")"',
            self.template,
        )
        self.assertIn(
            "TF_VAR_IMPORT_EXISTING_PLATFORM_CONNECTIVITY: "
            "${{ parameters.importExistingPlatformConnectivity }}",
            self.template,
        )
        self.assertIn(
            '-var "import_existing_platform_connectivity='
            '$TF_VAR_IMPORT_EXISTING_PLATFORM_CONNECTIVITY"',
            self.template,
        )


if __name__ == "__main__":
    unittest.main()
