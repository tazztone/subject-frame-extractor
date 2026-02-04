import sys
import pytest
from unittest.mock import patch
from core.config import Config
from app import parse_args

def test_config_defaults():
    config = Config()
    assert config.server_name == "127.0.0.1"
    assert config.server_port == 7860
    assert config.share is False
    assert config.auth is None
    assert config.ssl_keyfile is None
    assert config.ssl_certfile is None
    assert config.ssl_verify is True

def test_config_overrides():
    config = Config(
        server_name="0.0.0.0",
        server_port=9090,
        share=True,
        auth="user:pass",
        ssl_verify=False
    )
    assert config.server_name == "0.0.0.0"
    assert config.server_port == 9090
    assert config.share is True
    assert config.auth == "user:pass"
    assert config.ssl_verify is False

def test_parse_args():
    test_args = [
        "--server-name", "0.0.0.0",
        "--server-port", "8000",
        "--share",
        "--auth", "u:p",
        "--no-ssl-verify"
    ]
    with patch.object(sys, 'argv', ["app.py"] + test_args):
        args = parse_args()
        assert args.server_name == "0.0.0.0"
        assert args.server_port == 8000
        assert args.share is True
        assert args.auth == "u:p"
        assert args.ssl_verify is False

def test_parse_args_defaults():
    with patch.object(sys, 'argv', ["app.py"]):
        args = parse_args()
        assert args.server_name is None
        assert args.server_port is None
        assert args.share is None
        assert args.auth is None
        assert args.ssl_verify is None
