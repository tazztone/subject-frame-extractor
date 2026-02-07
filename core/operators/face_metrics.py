import math
import numpy as np
import mediapipe as mp
from core.operators import OperatorConfig, OperatorResult, OperatorContext, register_operator

def _get_face_data(ctx: OperatorContext) -> tuple[Optional[dict], Optional[np.ndarray]]:
    """Helper to get or compute face landmarks and blendshapes."""
    if "face_landmarker_result" in ctx.shared_data:
        return ctx.shared_data["face_landmarker_result"], ctx.shared_data.get("face_bbox")

    if not ctx.model_registry:
        return None, None

    # Load Face Landmarker via registry
    from core.managers import get_face_landmarker
    # We need landmarker_path. Usually derived from config.
    landmarker_url = getattr(ctx.config, "face_landmarker_url", None)
    if not landmarker_url:
        return None, None
        
    import os
    from pathlib import Path
    models_dir = getattr(ctx.config, "models_dir", "models")
    landmarker_path = Path(models_dir) / Path(landmarker_url).name
    
    if not landmarker_path.exists():
        return None, None

    try:
        landmarker = get_face_landmarker(str(landmarker_path), ctx.logger)
        
        # Determine face image (full thumb or crop)
        face_bbox = ctx.params.get("face_bbox")
        if face_bbox:
            x1, y1, x2, y2 = face_bbox
            face_img = ctx.image_rgb[y1:y2, x1:x2]
        else:
            face_img = ctx.image_rgb

        if not face_img.flags["C_CONTIGUOUS"]:
            face_img = np.ascontiguousarray(face_img, dtype=np.uint8)
        if face_img.dtype != np.uint8:
            face_img = face_img.astype(np.uint8)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_img)
        result = landmarker.detect(mp_image)
        
        ctx.shared_data["face_landmarker_result"] = result
        ctx.shared_data["face_bbox"] = face_bbox
        return result, face_bbox
    except Exception as e:
        if ctx.logger:
            ctx.logger.warning(f"Face landmarker execution failed: {e}")
        return None, None

@register_operator
class EyesOpenOperator:
    @property
    def config(self) -> OperatorConfig:
        return OperatorConfig(
            name="eyes_open",
            display_name="Eyes Open Score",
            category="face",
            description="Measures degree of eyes openness (0-100).",
            min_value=0.0,
            max_value=100.0,
            requires_face=True,
        )

    def execute(self, ctx: OperatorContext) -> OperatorResult:
        result, _ = _get_face_data(ctx)
        if not result or not result.face_blendshapes:
            return OperatorResult(metrics={}, warnings=["No face blendshapes detected"])
            
        blendshapes = {b.category_name: b.score for b in result.face_blendshapes[0]}
        
        blink_left = blendshapes.get("eyeBlinkLeft", 0.0)
        blink_right = blendshapes.get("eyeBlinkRight", 0.0)
        
        blink_prob = max(blink_left, blink_right)
        eyes_open = 1.0 - blink_prob
        
        return OperatorResult(metrics={
            "eyes_open": float(eyes_open), # Store 0-1 for normalized calc
            "eyes_open_score": eyes_open * 100.0,
            "blink_prob": blink_prob
        })


@register_operator
class FacePoseOperator:
    @property
    def config(self) -> OperatorConfig:
        return OperatorConfig(
            name="face_pose",
            display_name="Face Pose",
            category="face",
            description="Estimates face yaw, pitch, and roll.",
            min_value=-180.0,
            max_value=180.0,
            requires_face=True,
        )

    def execute(self, ctx: OperatorContext) -> OperatorResult:
        result, _ = _get_face_data(ctx)
        if not result or not result.facial_transformation_matrixes:
             return OperatorResult(metrics={}, warnings=["No face transformation matrix detected"])
        
        try:
             matrix = result.facial_transformation_matrixes[0]
             sy = math.sqrt(matrix[0, 0] * matrix[0, 0] + matrix[1, 0] * matrix[1, 0])
             singular = sy < 1e-6
             
             if not singular:
                 pitch = math.degrees(math.atan2(-matrix[2, 0], sy))
                 yaw = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
                 roll = math.degrees(math.atan2(matrix[2, 1], matrix[2, 2]))
             else:
                 pitch = math.degrees(math.atan2(-matrix[2, 0], sy))
                 yaw = 0.0
                 roll = 0.0
                 
             return OperatorResult(metrics={
                 "yaw": yaw,
                 "pitch": pitch,
                 "roll": roll
             })
             
        except Exception as e:
            return OperatorResult(error=str(e))
