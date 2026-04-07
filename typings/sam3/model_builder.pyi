from typing import Any, List, Optional

def build_sam3_predictor(
    checkpoint_path: Optional[str] = None,
    version: str = "sam3.1",
    compile: bool = False,
    use_fa3: bool = False,
    use_rope_real: bool = False,
    gpus_to_use: Optional[List[int]] = None,
) -> Any: ...
