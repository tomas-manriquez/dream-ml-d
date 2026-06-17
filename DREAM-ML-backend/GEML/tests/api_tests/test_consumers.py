# Copyright (C) 2025 Leonardo Espinoza Ortiz <leonardo.espinoza.o@usach.cl>
#
# This file is part of DREAM ML.
#
# DREAM ML is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DREAM ML is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DREAM ML. If not, see <https://www.gnu.org/licenses/>.

import json
import pytest
from unittest.mock import AsyncMock, Mock, patch
from api.consumers import ProgressConsumer


@pytest.fixture
def consumer():
    """Create a ProgressConsumer instance with mocked dependencies."""
    consumer = ProgressConsumer()
    consumer.channel_layer = AsyncMock()
    consumer.channel_name = "test_channel"
    consumer.accept = AsyncMock()
    consumer.send = AsyncMock()
    return consumer


class TestConnectionTesting:
    """Test cases for WebSocket connection functionality."""

    @pytest.mark.asyncio
    async def test_successful_websocket_connection(self, consumer):
        """
        Scenario 1: Successful WebSocket connection
        Given a WebSocket consumer instance
        When a client attempts to connect
        Then the channel should be added to "progreso_group" and the connection should be accepted
        """
        # Arrange
        consumer.channel_layer.group_add.return_value = None
        consumer.accept.return_value = None

        # Act
        await consumer.connect()

        # Assert
        consumer.channel_layer.group_add.assert_called_once_with(
            "progreso_group", "test_channel"
        )
        consumer.accept.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_with_channel_layer_failure(self, consumer):
        """
        Scenario 2: Connection with channel layer failure
        Given a WebSocket consumer instance and a failing channel layer
        When a client attempts to connect
        Then the connection process should handle the channel layer error appropriately
        """
        # Arrange
        consumer.channel_layer.group_add.side_effect = Exception("Channel layer error")

        # Act & Assert
        with pytest.raises(Exception, match="Channel layer error"):
            await consumer.connect()
        
        # Verify group_add was called even though it failed
        consumer.channel_layer.group_add.assert_called_once_with(
            "progreso_group", "test_channel"
        )
        # accept() should not be called if group_add fails
        consumer.accept.assert_not_called()


class TestDisconnectionTesting:
    """Test cases for WebSocket disconnection functionality."""

    @pytest.mark.asyncio
    async def test_successful_websocket_disconnection_normal_close(self, consumer):
        """
        Scenario 3: Successful WebSocket disconnection with normal close code
        Given a connected WebSocket consumer
        When the client disconnects with close code 1000 (normal closure)
        Then the channel should be removed from "progreso_group"
        """
        # Arrange
        close_code = 1000
        consumer.channel_layer.group_discard.return_value = None

        # Act
        await consumer.disconnect(close_code)

        # Assert
        consumer.channel_layer.group_discard.assert_called_once_with(
            "progreso_group", "test_channel"
        )

    @pytest.mark.asyncio
    async def test_disconnection_with_abnormal_close_code(self, consumer):
        """
        Scenario 4: Disconnection with abnormal close code
        Given a connected WebSocket consumer
        When the client disconnects with close code 1006 (abnormal closure)
        Then the channel should still be removed from "progreso_group"
        """
        # Arrange
        close_code = 1006
        consumer.channel_layer.group_discard.return_value = None

        # Act
        await consumer.disconnect(close_code)

        # Assert
        consumer.channel_layer.group_discard.assert_called_once_with(
            "progreso_group", "test_channel"
        )

    @pytest.mark.asyncio
    async def test_disconnection_with_channel_layer_failure(self, consumer):
        """
        Scenario 5: Disconnection with channel layer failure
        Given a connected WebSocket consumer and a failing channel layer
        When the client disconnects
        Then the disconnection should handle the channel layer error appropriately
        """
        # Arrange
        close_code = 1000
        consumer.channel_layer.group_discard.side_effect = Exception("Channel layer error")

        # Act & Assert
        with pytest.raises(Exception, match="Channel layer error"):
            await consumer.disconnect(close_code)
        
        # Verify group_discard was called even though it failed
        consumer.channel_layer.group_discard.assert_called_once_with(
            "progreso_group", "test_channel"
        )


class TestProgressMessageTesting:
    """Test cases for progress message sending functionality."""

    @pytest.mark.asyncio
    async def test_send_progress_with_both_step_and_status(self, consumer):
        """
        Scenario 6: Send progress with both step and status
        Given a connected WebSocket consumer
        When send_progress is called with event containing "step" and "status"
        Then a JSON message with both fields should be sent to the client
        """
        # Arrange
        event = {"step": "processing", "status": "in_progress"}
        expected_message = json.dumps({"step": "processing", "status": "in_progress"})

        # Act
        await consumer.send_progress(event)

        # Assert
        consumer.send.assert_called_once_with(text_data=expected_message)

    @pytest.mark.asyncio
    async def test_send_progress_with_missing_step(self, consumer):
        """
        Scenario 7: Send progress with missing step
        Given a connected WebSocket consumer
        When send_progress is called with event containing only "status"
        Then a JSON message with step=None and the provided status should be sent
        """
        # Arrange
        event = {"status": "completed"}
        expected_message = json.dumps({"step": None, "status": "completed"})

        # Act
        await consumer.send_progress(event)

        # Assert
        consumer.send.assert_called_once_with(text_data=expected_message)

    @pytest.mark.asyncio
    async def test_send_progress_with_missing_status(self, consumer):
        """
        Scenario 8: Send progress with missing status
        Given a connected WebSocket consumer
        When send_progress is called with event containing only "step"
        Then a JSON message with the provided step and status=None should be sent
        """
        # Arrange
        event = {"step": "validation"}
        expected_message = json.dumps({"step": "validation", "status": None})

        # Act
        await consumer.send_progress(event)

        # Assert
        consumer.send.assert_called_once_with(text_data=expected_message)

    @pytest.mark.asyncio
    async def test_send_progress_with_empty_event(self, consumer):
        """
        Scenario 9: Send progress with empty event
        Given a connected WebSocket consumer
        When send_progress is called with an empty event dict
        Then a JSON message with both step=None and status=None should be sent
        """
        # Arrange
        event = {}
        expected_message = json.dumps({"step": None, "status": None})

        # Act
        await consumer.send_progress(event)

        # Assert
        consumer.send.assert_called_once_with(text_data=expected_message)

    @pytest.mark.asyncio
    async def test_send_progress_with_non_serializable_data(self, consumer):
        """
        Scenario 10: Send progress with non-serializable data
        Given a connected WebSocket consumer
        When send_progress is called with event containing non-JSON-serializable objects
        Then the JSON serialization should handle the data appropriately (or raise appropriate error)
        """
        # Arrange
        # Create a non-serializable object (like a function or complex object)
        non_serializable_obj = lambda x: x  # Function objects are not JSON serializable
        event = {"step": non_serializable_obj, "status": "test"}

        # Act & Assert
        # The current implementation will try to serialize the non-serializable object
        # which should raise a TypeError from json.dumps()
        with pytest.raises(TypeError):
            await consumer.send_progress(event)

        # Verify that send was not called due to the JSON serialization error
        consumer.send.assert_not_called()


# Additional integration-style tests for edge cases
class TestEdgeCases:
    """Additional edge case tests for the consumer."""

    @pytest.mark.asyncio
    async def test_send_progress_with_none_event(self, consumer):
        """
        Test send_progress behavior when event is None
        This tests the robustness of the .get() method calls
        """
        # Arrange
        event = None
        
        # Act & Assert
        # This should raise an AttributeError because None doesn't have a .get() method
        with pytest.raises(AttributeError):
            await consumer.send_progress(event)

    @pytest.mark.asyncio 
    async def test_send_progress_with_extra_fields(self, consumer):
        """
        Test that extra fields in the event are ignored
        """
        # Arrange
        event = {
            "step": "processing", 
            "status": "active", 
            "extra_field": "ignored",
            "another_field": 123
        }
        expected_message = json.dumps({"step": "processing", "status": "active"})

        # Act
        await consumer.send_progress(event)

        # Assert
        consumer.send.assert_called_once_with(text_data=expected_message)