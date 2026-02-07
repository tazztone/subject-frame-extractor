import cv2
import numpy as np
from core.operators import OperatorConfig, OperatorResult, OperatorContext, register_operator
from core.operators.face_metrics import _get_face_data

@register_operator
class FaceSimilarityOperator:
    @property
    def config(self) -> OperatorConfig:
        return OperatorConfig(
            name="face_sim",
            display_name="Face Similarity",
            category="face",
            description="Computes similarity against a reference face embedding.",
            min_value=0.0,
            max_value=1.0,
            requires_face=True,
        )

    def execute(self, ctx: OperatorContext) -> OperatorResult:
        if not ctx.model_registry:
            return OperatorResult(error="Missing model registry")
            
        ref_emb = ctx.params.get("reference_embedding")
        if ref_emb is None:
            return OperatorResult(metrics={}, warnings=["Missing reference face embedding"])

        try:
            # We need the full face analyzer result to get embeddings
            # _get_face_data only gets landmarks. We need insightface here.
            # But wait, FaceLandmarker (MediaPipe) is different from FaceAnalysis (InsightFace).
            # The project uses InsightFace for similarity and MediaPipe for pose/blink.
            
            # Get face analyzer from registry
            from core.managers import get_face_analyzer
            face_model_name = getattr(ctx.config, "default_face_model_name", "buffalo_l")
            models_path = str(getattr(ctx.config, "models_dir", "models"))
            det_size = tuple(getattr(ctx.config, "model_face_analyzer_det_size", [640, 640]))
            
            analyzer = get_face_analyzer(
                model_name=face_model_name,
                models_path=models_path,
                det_size_tuple=det_size,
                logger=ctx.logger,
                model_registry=ctx.model_registry,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            # Run analyzer
            image_bgr = cv2.cvtColor(ctx.image_rgb, cv2.COLOR_RGB2BGR)
            faces = analyzer.get(image_bgr)
            
            if not faces:
                return OperatorResult(metrics={"face_sim": 0.0}, warnings=["No faces detected"])
                
            best_face = max(faces, key=lambda x: x.det_score)
            distance = 1 - np.dot(best_face.normed_embedding, ref_emb)
            similarity = 1.0 - float(distance)
            
            return OperatorResult(metrics={
                "face_sim": similarity,
                "face_conf": float(best_face.det_score)
            })
            
        except Exception as e:
            return OperatorResult(error=str(e))


@register_operator
class FaceProminenceOperator:
    @property
    def config(self) -> OperatorConfig:
        return OperatorConfig(
            name="face_prominence",
            display_name="Face Prominence",
            category="face",
            description="Measures how prominent (large/central) the face is.",
            min_value=0.0,
            max_value=100.0,
            requires_face=True,
        )

    def execute(self, ctx: OperatorContext) -> OperatorResult:
        # Re-use FaceSimilarity detections if they exist in shared_data
        # Actually, let's just use the analyzer again or cache results.
        # For now, let's assume we can run it or get it from ctx.params
        
        face_bbox = ctx.params.get("face_bbox")
        if not face_bbox:
             # Try to find it via analyzer if not passed
             return OperatorResult(metrics={"face_prominence_score": 0.0}, warnings=["No face bbox provided"])
             
        x1, y1, x2, y2 = face_bbox
        face_w, face_h = x2 - x1, y2 - y1
        face_area = face_w * face_h
        img_h, img_w = ctx.image_rgb.shape[:2]
        img_area = img_w * img_h
        
        area_pct = (face_area / img_area) * 100.0
        
        # Centrality
        face_cx, face_cy = (x1 + x2) / 2, (y1 + y2) / 2
        dist_from_center = math.sqrt(((face_cx - img_w/2)/img_w)**2 + ((face_cy - img_h/2)/img_h)**2)
        centrality = 1.0 - min(1.0, dist_from_center * 2) # 1.0 at center, 0.0 at edges
        
        # Weighted score
        prominence = (area_pct * 0.7) + (centrality * 30.0)
        
        return OperatorResult(metrics={"face_prominence_score": float(prominence)})
