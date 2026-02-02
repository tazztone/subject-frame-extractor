from pathlib import Path
from unittest.mock import MagicMock, patch, ANY
import json
import numpy as np

import pytest

from core.database import Database
from core.models import AnalysisParameters, Scene
from core.pipelines import (
    AnalysisPipeline, 
    PreAnalysisPipeline, 
    execute_pre_analysis,
    PreAnalysisEvent
)


class TestPipelinesExtended:
    @pytest.fixture
    def mock_logger(self):
        return MagicMock()

    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.seeding_iou_threshold = 0.5
        config.seeding_face_contain_score = 10
        config.seeding_confidence_score_multiplier = 1
        config.seeding_iou_bonus = 5
        config.seeding_balanced_score_weights = {"area": 1, "confidence": 1, "edge": 1}
        config.seeding_face_to_body_expansion_factors = [1.5, 3.0, 1.0]
        config.seeding_final_fallback_box = [0.25, 0.25, 0.75, 0.75]
        config.analysis_default_batch_size = 1
        config.retry_max_attempts = 1
        config.retry_backoff_seconds = (0.1,)
        config.downloads_dir = "downloads"
        config.model_face_analyzer_det_size = [640, 640]
        config.sharpness_base_scale = 1000
        config.edge_strength_base_scale = 100
        return config

    @pytest.fixture
    def mock_db(self):
        db = MagicMock(spec=Database)
        db.count_errors.return_value = 0
        return db

    @pytest.fixture
    def mock_params(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        return AnalysisParameters(
            source_path="video.mp4",
            video_path="video.mp4",
            output_folder=str(out),
            tracker_model_name="sam3",
            enable_face_filter=True,
            face_model_name="buffalo_l",
            face_ref_img_path="ref.jpg",
        )

    @pytest.fixture
    def mock_thumbnail_manager(self):
        return MagicMock()

    @pytest.fixture
    def pipeline(self, mock_params, mock_logger, mock_config, mock_db, mock_thumbnail_manager):
        mock_registry = MagicMock()
        mock_queue = MagicMock()
        mock_cancel = MagicMock()
        mock_cancel.is_set.return_value = False

        # Patch Database because AnalysisPipeline instantiates it in __init__
        with patch("core.pipelines.Database", return_value=mock_db):
            pipeline = AnalysisPipeline(
                config=mock_config,
                logger=mock_logger,
                params=mock_params,
                progress_queue=mock_queue,
                cancel_event=mock_cancel,
                thumbnail_manager=mock_thumbnail_manager,
                model_registry=mock_registry,
            )
        return pipeline

    @patch("core.pipelines.initialize_analysis_models")
    @patch("core.pipelines.SubjectMasker")
    def test_run_full_analysis_propagation(self, mock_masker_cls, mock_init_models, pipeline, mock_params):
        # Setup mocks
        mock_models = {
            "face_analyzer": MagicMock(),
            "ref_emb": MagicMock(),
            "face_landmarker": MagicMock(),
            "device": "cpu",
        }
        mock_init_models.return_value = mock_models

        mock_masker = mock_masker_cls.return_value
        # run_propagation returns a dict of metadata
        mock_masker.run_propagation.return_value = {"frame_0.png": {"mask_path": "path"}}

        # Use simple MagicMock for Scene to avoid spec/validation issues during testing
        mock_scene = MagicMock()
        mock_scene.shot_id = 1
        mock_scene.start_frame = 0
        mock_scene.end_frame = 10
        mock_scene.seed_result = MagicMock()

        scenes = [mock_scene]

        # Patch _process_reference_face to avoid file check
        with patch.object(pipeline, "_process_reference_face"):
            # Test run_full_analysis (which currently runs propagation for video)
            result = pipeline.run_full_analysis(scenes)

        # Verify
        if not result.get("done"):
            pytest.fail(f"Pipeline failed: {result}")

        mock_init_models.assert_called()
        mock_masker.run_propagation.assert_called()
        assert pipeline.mask_metadata == {"frame_0.png": {"mask_path": "path"}}

    @patch("core.pipelines.initialize_analysis_models")
    def test_run_analysis_only(self, mock_init_models, pipeline, mock_params):
        # Setup mocks
        mock_models = {
            "face_analyzer": MagicMock(),
            "ref_emb": MagicMock(),
            "face_landmarker": MagicMock(),
            "device": "cpu",
        }
        mock_init_models.return_value = mock_models

        mock_scene = MagicMock()
        mock_scene.shot_id = 1
        mock_scene.start_frame = 0
        mock_scene.end_frame = 10
        scenes = [mock_scene]

        # Mock _run_analysis_loop
        pipeline._run_analysis_loop = MagicMock()

        # Patch _process_reference_face to avoid file check
        with patch.object(pipeline, "_process_reference_face"):
            # Run
            result = pipeline.run_analysis_only(scenes)

        if not result.get("done"):
            pytest.fail(f"Pipeline failed: {result}")

        # Verify
        mock_init_models.assert_called()
        pipeline._run_analysis_loop.assert_called()
        pipeline.db.flush.assert_called()

    def test_cancellation_in_propagation(self, pipeline, mock_params):
        pipeline.cancel_event.is_set.return_value = True

        mock_scene = MagicMock()
        mock_scene.shot_id = 1
        scenes = [mock_scene]

        # Mock dependencies to reach cancellation check
        with patch(
            "core.pipelines.initialize_analysis_models",
            return_value={"face_analyzer": None, "ref_emb": None, "face_landmarker": None, "device": "cpu"},
        ):
            with patch("core.pipelines.SubjectMasker"):
                # Should return early or log cancellation
                result = pipeline.run_full_analysis(scenes)

        # Logic returns {"log": "Propagation cancelled.", "done": False} OR just loops.
        # If cancellation is checked inside the loop:
        # for scene in scenes: if cancel: break.
        # Then it proceeds.
        # Check specific return value in code:
        # if self.cancel_event.is_set(): return {"log": "Propagation cancelled.", "done": False}

        assert result["done"] is False
        assert "cancelled" in str(result).lower()


class TestPreAnalysisPipeline:
    @pytest.fixture
    def pre_pipeline(self, mock_params, mock_logger, mock_config, mock_thumbnail_manager):
        mock_registry = MagicMock()
        mock_queue = MagicMock()
        mock_cancel = MagicMock()
        mock_cancel.is_set.return_value = False

        return PreAnalysisPipeline(
            config=mock_config,
            logger=mock_logger,
            params=mock_params,
            progress_queue=mock_queue,
            cancel_event=mock_cancel,
            thumbnail_manager=mock_thumbnail_manager,
            model_registry=mock_registry,
        )

    @patch("core.pipelines.initialize_analysis_models")
    @patch("core.pipelines.SubjectMasker")
    @patch("core.pipelines.save_scene_seeds")
    @patch("PIL.Image.fromarray")
    def test_pre_analysis_run(self, mock_img_save, mock_save_seeds, mock_masker_cls, mock_init_models, pre_pipeline, tmp_path):
        # Setup mocks
        mock_init_models.return_value = {
            "face_analyzer": MagicMock(),
            "ref_emb": MagicMock(),
            "face_landmarker": MagicMock(),
            "device": "cpu",
        }
        
        mock_masker = mock_masker_cls.return_value
        mock_masker.frame_map = {1: "frame_000001.webp"}
        mock_masker.get_seed_for_frame.return_value = ([0, 0, 10, 10], {"score": 1.0})
        mock_masker.get_mask_for_bbox.return_value = np.zeros((100, 100), dtype=np.uint8)
        mock_masker.draw_bbox.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        pre_pipeline.thumbnail_manager.get.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        scene = Scene(shot_id=0, start_frame=0, end_frame=10)
        scene.best_frame = 1
        
        # Act
        result_scenes = pre_pipeline.run([scene])

        # Assert
        assert len(result_scenes) == 1
        assert result_scenes[0].seed_result["bbox"] == [0, 0, 10, 10]
        assert Path(result_scenes[0].preview_path).name == "scene_00000.jpg"
        mock_save_seeds.assert_called_once()


class TestExecutePreAnalysis:
    @patch("core.pipelines.PreAnalysisPipeline")
    @patch("core.pipelines._load_scenes")
    @patch("core.pipelines._initialize_pre_analysis_params")
    @patch("core.pipelines._handle_pre_analysis_uploads")
    def test_execute_pre_analysis_success(self, mock_handle_uploads, mock_init_params, mock_load_scenes, mock_pipeline_cls, tmp_path):
        # Setup
        mock_event = MagicMock(spec=PreAnalysisEvent)
        mock_event.model_dump.return_value = {}
        mock_event.face_ref_img_upload = None
        
        mock_handle_uploads.return_value = {}
        
        params = MagicMock(spec=AnalysisParameters)
        params.output_folder = str(tmp_path)
        params.video_path = "video.mp4"
        params.face_ref_img_path = None
        mock_init_params.return_value = (params, tmp_path)
        
        mock_load_scenes.return_value = [Scene(shot_id=0, start_frame=0, end_frame=10)]
        
        mock_pipeline = mock_pipeline_cls.return_value
        mock_pipeline.run.return_value = [Scene(shot_id=0, start_frame=0, end_frame=10)]

        # Act
        gen = execute_pre_analysis(
            event=mock_event,
            progress_queue=MagicMock(),
            cancel_event=MagicMock(),
            logger=MagicMock(),
            config=MagicMock(),
            thumbnail_manager=MagicMock(),
            cuda_available=False
        )
        results = list(gen)

        # Assert
        assert len(results) == 1
        assert results[0]["done"] is True
        assert "Pre-analysis complete" in results[0]["unified_log"]
