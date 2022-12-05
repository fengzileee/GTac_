#!/usr/bin/env python
import time
import sys
from gtac import GtacInterface, GtacVisualizer
import logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)

sensor = GtacInterface("/dev/ttyACM0", data_index=6)
sensor.start()

viz = GtacVisualizer(sensor)
viz.show()

sensor.stop()
