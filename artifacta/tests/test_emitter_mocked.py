"""Mocked integration tests for HTTP emitter"""

from unittest.mock import Mock, patch

from artifacta.emitter import HTTPEmitter


class TestHTTPEmitterMocked:
    """Test HTTP emitter with mocked requests"""

    @patch("artifacta.emitter.requests.Session")
    def test_emitter_initialization(self, mock_session_class):
        """Should initialize with health check"""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        emitter = HTTPEmitter(run_id="test-run", api_url="http://localhost:8000")

        assert emitter.run_id == "test-run"
        assert emitter.api_url == "http://localhost:8000"
        assert emitter.enabled is True
        mock_session.get.assert_called_once()

    @patch("artifacta.emitter.requests.Session")
    def test_emitter_handles_health_check_failure(self, mock_session_class):
        """Should disable on failed health check"""
        mock_session = Mock()
        mock_session.get.side_effect = Exception("Connection refused")
        mock_session_class.return_value = mock_session

        emitter = HTTPEmitter(run_id="test-run", api_url="http://localhost:8000")

        assert emitter.enabled is False

    @patch("artifacta.emitter.requests.Session")
    def test_emit_init_sends_post_request(self, mock_session_class):
        """Should send POST to /api/runs"""
        mock_session = Mock()
        mock_health_response = Mock()
        mock_health_response.status_code = 200
        mock_init_response = Mock()
        mock_init_response.status_code = 200

        mock_session.get.return_value = mock_health_response
        mock_session.post.return_value = mock_init_response
        mock_session_class.return_value = mock_session

        emitter = HTTPEmitter(run_id="test-run", api_url="http://localhost:8000")
        result = emitter.emit_init({"project": "test-proj", "config": {"lr": 0.001}})

        assert result is True
        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        assert "/api/runs" in call_args[0][0]

    @patch("artifacta.emitter.requests.Session")
    def test_emit_structured_data_sends_post(self, mock_session_class):
        """Should send POST to /api/runs/{run_id}/data"""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_session.get.return_value = mock_response
        mock_session.post.return_value = mock_response
        mock_session_class.return_value = mock_session

        emitter = HTTPEmitter(run_id="test-run", api_url="http://localhost:8000")
        result = emitter.emit_structured_data(
            {
                "name": "loss",
                "primitive_type": "series",
                "data": {"index": "epoch", "fields": {"loss": [1, 0.5]}},
            }
        )

        assert result is True
        # Get the last post call (first is health, second is this one)
        post_calls = list(mock_session.post.call_args_list)
        last_call = post_calls[-1]
        assert "/api/runs/test-run/data" in last_call[0][0]

    @patch("artifacta.emitter.requests.Session")
    def test_emit_artifact_sends_post(self, mock_session_class):
        """Should send POST to /api/artifacts"""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_session.get.return_value = mock_response
        mock_session.post.return_value = mock_response
        mock_session_class.return_value = mock_session

        emitter = HTTPEmitter(run_id="test-run", api_url="http://localhost:8000")
        result = emitter.emit_artifact(
            {
                "run_id": "test-run",
                "artifact_id": "art-123",
                "filepath": "/path/file.txt",
                "hash": "abc123",
            }
        )

        assert result is True
        post_calls = list(mock_session.post.call_args_list)
        last_call = post_calls[-1]
        assert "/api/artifacts" in last_call[0][0]

    @patch("artifacta.emitter.requests.Session")
    def test_emit_finish_sends_post(self, mock_session_class):
        """Should send POST to /api/runs/{run_id}/finish"""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_session.get.return_value = mock_response
        mock_session.post.return_value = mock_response
        mock_session_class.return_value = mock_session

        emitter = HTTPEmitter(run_id="test-run", api_url="http://localhost:8000")
        result = emitter.emit_finish(duration=120.5, exit_code=0)

        assert result is True
        post_calls = list(mock_session.post.call_args_list)
        last_call = post_calls[-1]
        assert "/api/runs/test-run/finish" in last_call[0][0]

        # Check request body
        request_json = last_call[1]["json"]
        assert request_json["duration"] == 120.5
        assert request_json["exit_code"] == 0

    @patch("artifacta.emitter.requests.Session")
    def test_emit_returns_false_when_disabled(self, mock_session_class):
        """Should return False for all emits when disabled"""
        mock_session = Mock()
        mock_session.get.side_effect = Exception("No connection")
        mock_session_class.return_value = mock_session

        emitter = HTTPEmitter(run_id="test-run", api_url="http://localhost:8000")

        assert emitter.emit_init({}) is False
        assert emitter.emit_structured_data({}) is False
        assert emitter.emit_artifact({}) is False
        assert emitter.emit_finish(1.0, 0) is False

    @patch("artifacta.emitter.requests.Session")
    def test_emit_handles_request_errors_gracefully(self, mock_session_class):
        """Should handle request errors without crashing"""
        mock_session = Mock()
        mock_health_response = Mock()
        mock_health_response.status_code = 200
        mock_session.get.return_value = mock_health_response

        # Simulate request failure
        mock_session.post.side_effect = Exception("Network error")
        mock_session_class.return_value = mock_session

        emitter = HTTPEmitter(run_id="test-run", api_url="http://localhost:8000")

        # Should not raise, returns False
        result = emitter.emit_structured_data({"test": "data"})
        assert result is False

    @patch("artifacta.emitter.requests.Session")
    def test_emitter_uses_custom_api_url(self, mock_session_class):
        """Should use custom API URL"""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        custom_url = "http://custom-server:9000"
        emitter = HTTPEmitter(run_id="test", api_url=custom_url)

        assert emitter.api_url == custom_url
        # Health check should use custom URL
        health_call = mock_session.get.call_args[0][0]
        assert custom_url in health_call
