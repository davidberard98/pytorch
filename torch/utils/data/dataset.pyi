# This base template ("dataset.pyi.in") is generated from mypy stubgen with minimal editing for code injection
# The output file will be "dataset.pyi".
# Note that, for mypy, .pyi file takes precedent over .py file, such that we must define the interface for other
# classes/objects here, even though we are not injecting extra code into them at the moment.

from ... import Generator as Generator, Tensor as Tensor
from torch import default_generator as default_generator, randperm as randperm
from torch.utils.data._typing import _DataPipeMeta
from typing import Any, Callable, Dict, Generic, Iterable, Iterator, List, Optional, Sequence, Tuple, TypeVar

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')
UNTRACABLE_DATAFRAME_PIPES: Any


class DataChunk(list, Generic[T]):
    items: Any = ...
    def __init__(self, items: Any) -> None: ...
    def as_str(self, indent: str = ...): ...
    def __iter__(self) -> Iterator[T]: ...
    def raw_iterator(self) -> T: ...

class Dataset(Generic[T_co]):
    functions: Dict[str, Callable] = ...
    def __getitem__(self, index: Any) -> T_co: ...
    def __add__(self, other: Dataset[T_co]) -> ConcatDataset[T_co]: ...
    def __getattr__(self, attribute_name: Any): ...
    @classmethod
    def register_function(cls, function_name: Any, function: Any) -> None: ...
    @classmethod
    def register_datapipe_as_function(cls, function_name: Any, cls_to_register: Any, enable_df_api_tracing: bool = ...): ...

MapDataPipe = Dataset

class IterableDataset(Dataset[T_co], metaclass=_DataPipeMeta):
    functions: Dict[str, Callable] = ...
    reduce_ex_hook: Optional[Callable] = ...
    def __iter__(self) -> Iterator[T_co]: ...
    def __add__(self, other: Dataset[T_co]) -> Any: ...
    def __getattr__(self, attribute_name: Any): ...
    def __reduce_ex__(self, *args: Any, **kwargs: Any): ...
    @classmethod
    def set_reduce_ex_hook(cls, hook_fn: Any) -> None: ...
    # Functional form of 'BatcherIterDataPipe'
    def batch(self, batch_size: int, drop_last: bool = False, unbatch_level: int = 0, wrapper_class=DataChunk): ...
    # Functional form of 'CollatorIterDataPipe'
    def collate(self, collate_fn: Callable= ..., fn_args: Optional[Tuple] = None, fn_kwargs: Optional[Dict] = None): ...
    # Functional form of 'ConcaterIterDataPipe'
    def concat(self, *datapipes: IterDataPipe): ...
    # Functional form of 'DemultiplexerIterDataPipe'
    def demux(self, num_instances: int, classifier_fn: Callable[[T_co], int], drop_none: bool = False, buffer_size: int = 1000): ...
    # Functional form of 'FilterIterDataPipe'
    def filter(self, filter_fn: Callable, fn_args: Optional[Tuple] = None, fn_kwargs: Optional[Dict] = None, drop_empty_batches: bool = True, nesting_level: int = 0): ...
    # Functional form of 'ForkerIterDataPipe'
    def fork(self, num_instances: int, buffer_size: int = 1000): ...
    # Functional form of 'GrouperIterDataPipe'
    def groupby(self, group_key_fn: Callable, *, buffer_size: int = 10000, group_size: Optional[int] = None, unbatch_level: int = 0, guaranteed_group_size: Optional[int] = None, drop_remaining: bool = False): ...
    # Functional form of 'MapperIterDataPipe'
    def map(self, fn: Callable, input_col=None, output_col=None, *, fn_args: Optional[Tuple] = None, fn_kwargs: Optional[Dict] = None, nesting_level: int = 0): ...
    # Functional form of 'MultiplexerIterDataPipe'
    def mux(self, *datapipes): ...
    # Functional form of 'RoutedDecoderIterDataPipe'
    def decode(self, *handlers: Callable, key_fn: Callable= ...): ...
    # Functional form of 'ShardingFilterIterDataPipe'
    def sharding_filter(self): ...
    # Functional form of 'ShufflerIterDataPipe'
    def shuffle(self, *, buffer_size: int = 10000, unbatch_level: int = 0): ...
    # Functional form of 'UnBatcherIterDataPipe'
    def unbatch(self, unbatch_level: int = 1): ...
    # Functional form of 'ZipperIterDataPipe'
    def zip(self, *datapipes: IterDataPipe): ...

IterDataPipe = IterableDataset

class DFIterDataPipe(IterableDataset): ...

class TensorDataset(Dataset[Tuple[Tensor, ...]]):
    tensors: Tuple[Tensor, ...]
    def __init__(self, *tensors: Tensor) -> None: ...
    def __getitem__(self, index: Any): ...
    def __len__(self): ...

class ConcatDataset(Dataset[T_co]):
    datasets: List[Dataset[T_co]]
    cumulative_sizes: List[int]
    @staticmethod
    def cumsum(sequence: Any): ...
    def __init__(self, datasets: Iterable[Dataset]) -> None: ...
    def __len__(self): ...
    def __getitem__(self, idx: Any): ...
    @property
    def cummulative_sizes(self): ...

class ChainDataset(IterableDataset):
    datasets: Any = ...
    def __init__(self, datasets: Iterable[Dataset]) -> None: ...
    def __iter__(self) -> Any: ...
    def __len__(self): ...

class Subset(Dataset[T_co]):
    dataset: Dataset[T_co]
    indices: Sequence[int]
    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int]) -> None: ...
    def __getitem__(self, idx: Any): ...
    def __len__(self): ...

def random_split(dataset: Dataset[T], lengths: Sequence[int], generator: Optional[Generator]=...) -> List[Subset[T]]: ...
