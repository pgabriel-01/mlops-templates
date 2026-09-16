"""Create or update an Azure ML compute cluster."""

import argparse

from azure.ai.ml.entities import AmlCompute

from aml_client import add_workspace_arguments, create_ml_client, wait_for_poller


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_workspace_arguments(parser)
    parser.add_argument("--cluster_name", required=True)
    parser.add_argument("--size", required=True)
    parser.add_argument("--min_instances", type=int, required=True)
    parser.add_argument("--max_instances", type=int, required=True)
    parser.add_argument("--cluster_tier")
    return parser.parse_args()


def run(args: argparse.Namespace):
    ml_client = create_ml_client(args)
    cluster = AmlCompute(
        name=args.cluster_name,
        size=args.size,
        min_instances=args.min_instances,
        max_instances=args.max_instances,
        tier=args.cluster_tier,
    )
    return wait_for_poller(ml_client.compute.begin_create_or_update(cluster))


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
