import math
import numpy as np
from core.operators import OperatorConfig, OperatorResult, OperatorContext, register_operator

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
        blendshapes = ctx.params.get("face_blendshapes")
        if not blendshapes:
            return OperatorResult(
                metrics={}, 
                warnings=["Missing face_blendshapes data"]
            )
            
        # Logic from core/models.py lines 152-155
        # 1.0 - max(eyeBlinkLeft, eyeBlinkRight)
        
        blink_left = blendshapes.get("eyeBlinkLeft", 0.0)
        blink_right = blendshapes.get("eyeBlinkRight", 0.0)
        
        blink_prob = max(blink_left, blink_right)
        eyes_open = 1.0 - blink_prob
        
        return OperatorResult(metrics={
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
        matrix = ctx.params.get("face_matrix")
        if matrix is None:
             return OperatorResult(
                metrics={}, 
                warnings=["Missing face_matrix data"]
            )
        
        # Logic from core/models.py lines 163-177
        # matrix is 4x4 numpy array (or similar indexable)
        try:
             # Ensure matrix is accessible
             # Check shape? Assuming proper matrix passed.
             
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
