from core.models import AnalysisParameters
from core.config import Config
from core.logger import AppLogger

config = Config()
logger = AppLogger(config)
p = AnalysisParameters.from_ui(logger, config)
print("quality:", p.compute_quality_score)
print("phash:", p.compute_phash)
print("contrast:", p.compute_contrast)
