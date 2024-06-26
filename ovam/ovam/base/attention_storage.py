"""Class in charge of storing the hidden states of a block.

The class OnlineAttentionStorage allows a simple version that 
stores all the hidden states in memory. The AttentionStorage
class is a generic class that can be used to implement more
complex storage classes.

"""
from typing import TYPE_CHECKING, Iterable, Optional, List

if TYPE_CHECKING:
    import torch

__all__ = ["AttentionStorage", "OnlineAttentionStorage"]


class AttentionStorage:
    """Generic class for storing hidden states of upsample/downsample block.

    Attributes
    ----------
    name: str
        The name of the block in the UNet.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name

    def store(self, hidden_states: "torch.Tensor") -> None:
        """Stores the hidden states.

        Arguments
        ---------
        hidden_states: List[torch.Tensor]
            The hidden states of a block generated by an image. The
            hidden states are stored in the order they are passed.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """Returns the number of images stored"""
        raise NotImplementedError

    def __getitem__(self, idx: int) -> "torch.Tensor":
        """Returns the hidden state at the given index."""
        raise NotImplementedError

    def __iter__(self) -> Iterable["torch.Tensor"]:
        """Returns an iterator over the stored hidden states."""
        for i in range(len(self)):
            yield self[i]

    def clear(self) -> None:
        """Clears the stored hidden states."""
        raise NotImplementedError


class OnlineAttentionStorage(AttentionStorage):
    """Class to store the hidden states in memory.

        Attributes
    ----------
    block_name: str
        The name of the block in the UNet.
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.hidden_states: List["torch.Tensor"] = []

    def store(self, hidden_states: "torch.Tensor") -> None:
        """Stores the hidden states.

        Arguments
        ---------
        hidden_states: List[torch.Tensor]
            The hidden states of a block generated by an image. The
            hidden states are stored in the order they are passed.
        """
        self.hidden_states.append(hidden_states)

    def __len__(self) -> int:
        return len(self.hidden_states)

    def __getitem__(self, idx: int) -> "torch.Tensor":
        return self.hidden_states[idx]

    def clear(self) -> None:
        """Clears the stored hidden states."""
        self.hidden_states.clear()
