# Decision 0010: Verify Before Removing A Pin

Status: active

## Decision

Before relaxing a validator, deleting a literal or removing a seemingly unused
artifact:

1. inspect the producer;
2. inspect every consumer and guard;
3. distinguish policy, measured value and derived relationship;
4. check the relevant spec and decision;
5. prove the proposed failure path, not only the success path.

Delete a literal only when it provides no live capability, evidence,
compatibility or fail-closed protection. A green test suite proves internal
consistency, not that the premise for a change is correct.

## Consequences

- Prefer derived invariant checks over duplicated constants.
- Preserve deliberate exclusions and consumed-authority guards.
- Remove weaker redundant tests when one stronger test catches the same unique
  failure.
