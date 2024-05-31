import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)-8s: %(levelname)-8s %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)
