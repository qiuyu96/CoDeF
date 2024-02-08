from .distributed_weighted_sampler import DistributedWeightedSampler
from .video_dataset import VideoDataset

dataset_dict = {"video": VideoDataset}

custom_sampler_dict = {"weighted": DistributedWeightedSampler}
