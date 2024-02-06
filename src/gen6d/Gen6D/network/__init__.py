from .detector import Detector
from .refiner import VolumeRefiner
from .selector import ViewpointSelector

name2network={
    'refiner': VolumeRefiner,
    'detector': Detector,
    'selector': ViewpointSelector,
}