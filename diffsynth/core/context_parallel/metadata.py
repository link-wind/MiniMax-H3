from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Iterable, Sequence


def split_sequence_indices(seq_len: int, world_size: int) -> list[tuple[int, int]]:
    """Return contiguous, roughly balanced global spans for each CP rank."""
    if seq_len < 0:
        raise ValueError(f"seq_len must be non-negative, got {seq_len}")
    if world_size < 1:
        raise ValueError(f"world_size must be positive, got {world_size}")
    return [
        (seq_len * rank // world_size, seq_len * (rank + 1) // world_size)
        for rank in range(world_size)
    ]


@dataclass(frozen=True)
class PackedAttentionMetadata:
    """Global packed-sequence metadata used by varlen attention."""

    cu_seqlens: tuple[int, ...]
    seq_len: int

    def __post_init__(self) -> None:
        cu = tuple(int(x) for x in self.cu_seqlens)
        seq_len = int(self.seq_len)
        if not cu or cu[0] != 0:
            raise ValueError(f"cu_seqlens must start at 0, got {cu}")
        if any(b < a for a, b in zip(cu, cu[1:])):
            raise ValueError(f"cu_seqlens must be non-decreasing, got {cu}")
        if cu[-1] > seq_len:
            raise ValueError(f"cu_seqlens end {cu[-1]} exceeds seq_len {seq_len}")
        object.__setattr__(self, "cu_seqlens", cu)
        object.__setattr__(self, "seq_len", seq_len)

    @classmethod
    def from_cu_seqlens(
        cls, cu_seqlens: Sequence[int], seq_len: int | None = None
    ) -> "PackedAttentionMetadata":
        cu = tuple(int(x) for x in cu_seqlens)
        return cls(cu, seq_len if seq_len is not None else cu[-1])

    def segment_id_at(self, position: int) -> int:
        if position < 0 or position >= self.seq_len:
            raise IndexError(f"position {position} outside [0, {self.seq_len})")
        return max(0, bisect_right(self.cu_seqlens, position) - 1)

    def segment_ids_for_positions(self, positions: Iterable[int]) -> tuple[int, ...]:
        return tuple(self.segment_id_at(int(position)) for position in positions)

    def segment_ids_for_span(self, start: int, end: int) -> tuple[int, ...]:
        return self.segment_ids_for_positions(range(start, end))


@dataclass(frozen=True)
class PackedShardMetadata:
    """Local view of global packed metadata for one CP rank."""

    metadata: PackedAttentionMetadata
    rank: int
    world_size: int
    chunk_spans: tuple[tuple[int, int], ...] | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.rank < self.world_size:
            raise ValueError(
                f"rank {self.rank} outside [0, {self.world_size})"
            )
        if self.chunk_spans is None:
            spans = split_sequence_indices(self.metadata.seq_len, self.world_size)
        else:
            spans = [
                (int(start), int(end)) for start, end in self.chunk_spans
            ]
            if len(spans) != self.world_size:
                raise ValueError(
                    f"chunk_spans length {len(spans)} != world_size {self.world_size}"
                )
            expected = 0
            for start, end in spans:
                if start != expected or end < start:
                    raise ValueError(f"chunk_spans are not contiguous, got {spans}")
                expected = end
            if expected != self.metadata.seq_len:
                raise ValueError(
                    f"chunk_spans end {expected} != seq_len {self.metadata.seq_len}"
                )
        object.__setattr__(self, "chunk_spans", tuple(spans))

    @property
    def local_span(self) -> tuple[int, int]:
        return self.chunk_spans[self.rank]

    @property
    def local_start(self) -> int:
        return self.local_span[0]

    @property
    def local_end(self) -> int:
        return self.local_span[1]

    @property
    def local_global_positions(self) -> tuple[int, ...]:
        return tuple(range(self.local_start, self.local_end))

    @property
    def local_segment_ids(self) -> tuple[int, ...]:
        return self.metadata.segment_ids_for_span(self.local_start, self.local_end)

    def chunk_span(self, chunk_rank: int) -> tuple[int, int]:
        if not 0 <= chunk_rank < self.world_size:
            raise IndexError(f"chunk_rank {chunk_rank} outside ring")
        return self.chunk_spans[chunk_rank]

    def segment_id_at_global_position(self, position: int) -> int:
        return self.metadata.segment_id_at(position)

    def segment_ids_for_chunk(self, chunk_rank: int) -> tuple[int, ...]:
        start, end = self.chunk_span(chunk_rank)
        return self.metadata.segment_ids_for_span(start, end)
