from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from gpu_scheduler_lab.queues.model import QueueSpec, ResourceVector

DEFAULT_QUEUE = "root/default"


class QueueHierarchy:
    def __init__(self, queues: Iterable[QueueSpec] = ()) -> None:
        supplied = list(queues)
        ids = [queue.id for queue in supplied]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate queue IDs are not allowed")
        specs = {queue.id: queue for queue in supplied}
        if "root" not in specs:
            specs["root"] = QueueSpec("root", None)
        if DEFAULT_QUEUE not in specs:
            specs[DEFAULT_QUEUE] = QueueSpec(DEFAULT_QUEUE, "root")
        if specs["root"].parent is not None:
            raise ValueError("root queue must not have a parent")
        for queue in specs.values():
            if queue.id != "root" and not queue.parent:
                raise ValueError(f"queue {queue.id} must declare a parent")
            if queue.parent is not None and queue.parent not in specs:
                raise ValueError(f"queue {queue.id} has missing parent {queue.parent}")
        self.specs = specs
        self.children: dict[str, list[str]] = defaultdict(list)
        for queue in specs.values():
            if queue.parent is not None:
                self.children[queue.parent].append(queue.id)
        for children in self.children.values():
            children.sort()
        self._validate_acyclic()
        self._validate_child_guarantees()

    def _validate_acyclic(self) -> None:
        for queue_id in self.specs:
            seen: set[str] = set()
            current: str | None = queue_id
            while current is not None:
                if current in seen:
                    raise ValueError(f"queue hierarchy contains a cycle at {current}")
                seen.add(current)
                current = self.specs[current].parent

    def _validate_child_guarantees(self) -> None:
        for parent_id, child_ids in self.children.items():
            parent = self.specs[parent_id]
            if parent.limit is None:
                continue
            aggregate = ResourceVector()
            for child_id in child_ids:
                aggregate += self.specs[child_id].guaranteed
            if not aggregate.fits_within(parent.limit):
                raise ValueError(f"child guarantees exceed parent {parent_id} limit")

    def ancestors(self, queue_id: str, *, include_self: bool = True) -> tuple[str, ...]:
        if queue_id not in self.specs:
            raise KeyError(queue_id)
        result: list[str] = []
        current: str | None = queue_id if include_self else self.specs[queue_id].parent
        while current is not None:
            result.append(current)
            current = self.specs[current].parent
        return tuple(result)

    def aggregate_usage(self, direct_usage: dict[str, ResourceVector]) -> dict[str, ResourceVector]:
        aggregate = {queue_id: ResourceVector() for queue_id in self.specs}
        for queue_id, usage in direct_usage.items():
            for ancestor in self.ancestors(queue_id):
                aggregate[ancestor] = aggregate[ancestor] + usage
        return aggregate

    def can_allocate(
        self,
        queue_id: str,
        demand: ResourceVector,
        direct_usage: dict[str, ResourceVector],
        *,
        borrowing: bool,
        aggregate_usage: dict[str, ResourceVector] | None = None,
    ) -> bool:
        aggregate = (
            self.aggregate_usage(direct_usage) if aggregate_usage is None else aggregate_usage
        )
        for ancestor_id in self.ancestors(queue_id):
            spec = self.specs[ancestor_id]
            projected = aggregate[ancestor_id] + demand
            if not projected.fits_within(spec.limit):
                return False
        queue = self.specs[queue_id]
        projected_queue = aggregate[queue_id] + demand
        if projected_queue.fits_within(queue.guaranteed):
            return True
        return borrowing and queue.borrowing_enabled

    def leaves(self) -> tuple[str, ...]:
        return tuple(sorted(queue_id for queue_id in self.specs if not self.children[queue_id]))

    def to_list(self) -> list[dict[str, object]]:
        return [self.specs[queue_id].to_dict() for queue_id in sorted(self.specs)]
