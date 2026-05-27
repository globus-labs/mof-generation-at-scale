"""Smoke-bootstrap + topic cleanup for the Octopus (AWS MSK) backend.

Not a pytest test (no test_ prefix, no assertions) — a sys.exit script that
walks delete_user → create_key → list_namespaces+delete_topic against the
caller's Octopus namespace. Three use cases:

    python tests/smoke_octopus.py                       # 1, 2
    python tests/smoke_octopus.py --prefix mofa_abc123  # 3
    python tests/smoke_octopus.py --no-rotate-keys --prefix mofa_abc123  # 3, idempotent

1. **First-time token setup** (envs/chameleon-stream.md §5): the create_key
   call triggers an interactive Globus login on first use and caches tokens
   at ~/.diaspora/storage.db. Subsequent runs are non-interactive.

2. **Lingering-topic cleanup** (envs/chameleon-stream.md §7): wipes every topic
   the user owns. Use when stale topics from a crashed `diaspora.py octopus`
   run accumulate and librdkafka starts failing with "Unable to create
   broker thread".

3. **Per-run cleanup** (called from bin/run-cloud-vm.sh): with --prefix,
   delete only topics whose name starts with that prefix. Pair with
   --no-rotate-keys to skip the destructive delete_user + create_key dance
   so post-workflow cleanup doesn't disturb the user's AWS credentials.

Destructive by default: rotates AWS access/secret keys and deletes every
topic in the caller's namespace. On a fresh account both are no-ops.
"""
import argparse
import logging
import sys

from diaspora_event_sdk import Client

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prefix", default=None,
                        help="If set, only delete topics whose name starts with this prefix. "
                             "Otherwise every topic in the caller's namespaces is deleted.")
    parser.add_argument("--no-rotate-keys", action="store_true",
                        help="Skip the delete_user + create_key step. Use after a successful "
                             "workflow run when you only want to clean a single run's topics "
                             "(--prefix) without invalidating cached AWS credentials.")
    args = parser.parse_args()

    c = Client()

    if not args.no_rotate_keys:
        # delete_user nukes the IAM user + namespace. Idempotent — on first run
        # (no user yet) the server still returns a non-error status.
        log.info("delete_user: %s", c.delete_user())

        # create_key re-provisions a fresh access/secret pair and creates the
        # user implicitly if missing. First call triggers interactive Globus auth.
        key_result = c.create_key()
        log.info("create_key: access=%s...", key_result["access_key"][:6])

    namespaces = c.list_namespaces().get("namespaces", {})
    log.info("list_namespaces: %s", namespaces)
    deleted = 0
    for ns, topics in namespaces.items():
        for topic in topics:
            if args.prefix is not None and not topic.startswith(args.prefix):
                continue
            log.info("delete_topic ns=%s topic=%s: %s", ns, topic, c.delete_topic(topic))
            deleted += 1

    after = c.list_namespaces().get("namespaces", {})
    if args.prefix is None:
        remaining = sum(len(t) for t in after.values())
    else:
        remaining = sum(1 for ts in after.values() for t in ts if t.startswith(args.prefix))
    log.info("deleted=%d remaining=%d", deleted, remaining)
    return 0 if remaining == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
