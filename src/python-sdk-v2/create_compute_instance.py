"""Create or update an Azure ML compute instance."""

import argparse

from azure.ai.ml.constants import ManagedServiceIdentityType
from azure.ai.ml.entities import (
    ComputeInstance,
    IdentityConfiguration,
    ManagedIdentityConfiguration,
)

from aml_client import add_workspace_arguments, create_ml_client, wait_for_poller


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_workspace_arguments(parser)
    parser.add_argument("--instance_name", required=True)
    parser.add_argument("--size", required=True)
    parser.add_argument("--location")
    parser.add_argument("--description")
    parser.add_argument(
        "--identity_type",
        choices=(
            ManagedServiceIdentityType.SYSTEM_ASSIGNED,
            ManagedServiceIdentityType.USER_ASSIGNED,
        ),
    )
    parser.add_argument("--user_assigned_identity")
    return parser.parse_args()


def run(args: argparse.Namespace):
    ml_client = create_ml_client(args)
    identity = None
    if args.identity_type == ManagedServiceIdentityType.SYSTEM_ASSIGNED:
        identity = IdentityConfiguration(
            type=ManagedServiceIdentityType.SYSTEM_ASSIGNED
        )
    elif args.identity_type == ManagedServiceIdentityType.USER_ASSIGNED:
        if not args.user_assigned_identity:
            raise ValueError(
                "--user_assigned_identity is required for a user-assigned identity"
            )
        managed_identity = ManagedIdentityConfiguration(
            resource_id=args.user_assigned_identity
        )
        identity = IdentityConfiguration(
            type=ManagedServiceIdentityType.USER_ASSIGNED,
            user_assigned_identities=[managed_identity],
        )

    instance = ComputeInstance(
        name=args.instance_name,
        size=args.size,
        location=args.location,
        description=args.description,
        identity=identity,
    )
    return wait_for_poller(
        ml_client.compute.begin_create_or_update(instance)
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
