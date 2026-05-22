"""Smoke-bootstrap + topic cleanup for the Octopus (AWS MSK) backend.

Not a pytest test (no test_ prefix, no assertions) — a sys.exit script that
walks delete_user → create_key → list_namespaces+delete_topic against the
caller's Octopus namespace. One command, two use cases:

    python tests/smoke_octopus.py

1. **First-time token setup** (envs/polaris-stream.md §5): the create_key
   call triggers an interactive Globus login on first use and caches tokens
   at ~/.diaspora/storage.db. Subsequent runs are non-interactive.

2. **Lingering-topic cleanup** (envs/polaris-stream.md §7): wipes every topic
   the user owns. Use when stale topics from a crashed `diaspora.py octopus`
   run accumulate and librdkafka starts failing with "Unable to create
   broker thread".

Destructive: rotates AWS access/secret keys and deletes every topic in the
caller's namespace. On a fresh account both are no-ops.
"""
import logging
import sys

from diaspora_event_sdk import Client

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def main() -> int:
    c = Client()

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
            log.info("delete_topic ns=%s topic=%s: %s", ns, topic, c.delete_topic(topic))
            deleted += 1

    after = c.list_namespaces().get("namespaces", {})
    remaining = sum(len(t) for t in after.values())
    log.info("deleted=%d remaining=%d", deleted, remaining)
    return 0 if remaining == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
