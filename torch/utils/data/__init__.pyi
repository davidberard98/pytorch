# These imports are necessary for functional DataPipe code completion
# Full set of possible imports
from .sampler import (
    BatchSampler as BatchSampler,
    RandomSampler as RandomSampler,
    Sampler as Sampler,
    SequentialSampler as SequentialSampler,
    SubsetRandomSampler as SubsetRandomSampler,
    WeightedRandomSampler as WeightedRandomSampler,
)
from .dataset import (
    ChainDataset as ChainDataset,
    ConcatDataset as ConcatDataset,
    DataChunk as DataChunk,
    Dataset as Dataset,
    Dataset as MapDataPipe,
    DFIterDataPipe as DFIterDataPipe,
    IterableDataset as IterableDataset,
    IterableDataset as IterDataPipe,
    Subset as Subset,
    TensorDataset as TensorDataset,
    random_split as random_split,
)
from .dataloader import (
    DataLoader as DataLoader,
    _DatasetKind as _DatasetKind,
    get_worker_info as get_worker_info,
)
from .distributed import DistributedSampler as DistributedSampler
from ._decorator import (
    argument_validation as argument_validation,
    functional_datapipe as functional_datapipe,
    guaranteed_datapipes_determinism as guaranteed_datapipes_determinism,
    non_deterministic as non_deterministic,
    runtime_validation as runtime_validation,
    runtime_validation_disabled as runtime_validation_disabled,
)
from .dataloader_experimental import DataLoader2 as DataLoader2
from . import communication as communication

# Full set of possible imports
# from torch.utils.data.sampler import (
#     BatchSampler as BatchSampler,
#     RandomSampler as RandomSampler,
#     Sampler as Sampler,
#     SequentialSampler as SequentialSampler,
#     SubsetRandomSampler as SubsetRandomSampler,
#     WeightedRandomSampler as WeightedRandomSampler,
# )
# from torch.utils.data.dataset import (
#     ChainDataset as ChainDataset,
#     ConcatDataset as ConcatDataset,
#     DataChunk as DataChunk,
#     Dataset as Dataset,
#     Dataset as MapDataPipe,
#     DFIterDataPipe as DFIterDataPipe,
#     IterableDataset as IterableDataset,
#     IterableDataset as IterDataPipe,
#     Subset as Subset,
#     TensorDataset as TensorDataset,
#     random_split as random_split,
# )
# from torch.utils.data.dataloader import (
#     DataLoader as DataLoader,
#     _DatasetKind as _DatasetKind,
#     get_worker_info as get_worker_info,
# )
# from torch.utils.data.distributed import DistributedSampler as DistributedSampler
# from torch.utils.data._decorator import (
#     argument_validation as argument_validation,
#     functional_datapipe as functional_datapipe,
#     guaranteed_datapipes_determinism as guaranteed_datapipes_determinism,
#     non_deterministic as non_deterministic,
#     runtime_validation as runtime_validation,
#     runtime_validation_disabled as runtime_validation_disabled,
# )
# from torch.utils.data.dataloader_experimental import DataLoader2 as DataLoader2
# from torch.utils.data import communication as communication
