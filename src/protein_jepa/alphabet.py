from dataclasses import dataclass


@dataclass(frozen=True)
class ProteinAlphabet:
    pad_token: str = "<pad>"
    mask_token: str = "<mask>"
    unk_token: str = "X"
    residues: str = "ACDEFGHIKLMNPQRSTVWY"

    def __post_init__(self) -> None:
        tokens = [self.pad_token, self.mask_token, self.unk_token, *self.residues]
        object.__setattr__(self, "tokens", tuple(tokens))
        object.__setattr__(self, "token_to_id", {token: i for i, token in enumerate(tokens)})
        object.__setattr__(self, "id_to_token", {i: token for i, token in enumerate(tokens)})

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def mask_id(self) -> int:
        return self.token_to_id[self.mask_token]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.unk_token]

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def clean(self, sequence: str) -> str:
        allowed = set(self.residues)
        return "".join(residue if residue in allowed else self.unk_token for residue in sequence.upper())

    def encode(self, sequence: str, max_length: int | None = None) -> list[int]:
        cleaned = self.clean(sequence)
        if max_length is not None:
            cleaned = cleaned[:max_length]
        return [self.token_to_id.get(residue, self.unk_id) for residue in cleaned]

    def decode(self, token_ids: list[int]) -> str:
        residues = []
        for token_id in token_ids:
            token = self.id_to_token.get(int(token_id), self.unk_token)
            if token in {self.pad_token, self.mask_token}:
                continue
            residues.append(token)
        return "".join(residues)

