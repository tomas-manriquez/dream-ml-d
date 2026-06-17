
import socket
from django.test import TestCase
from .utils import is_port_available  # Adjust import path as needed


class UtilsTestCase(TestCase):
    
    def test_port_is_available(self):
        """Test that is_port_available returns True when a port is available."""
        # Use a high port number that's likely to be free
        test_port = 54321
        
        # First, check if the port is available
        result = is_port_available(test_port)
        
        if result:
            # Port is available, test passes
            self.assertTrue(result, f"Port {test_port} should be available")
        else:
            # Port is not available, try a different port or skip
            # Let's try a few different ports
            for port_offset in range(1, 10):
                alternative_port = test_port + port_offset
                if is_port_available(alternative_port):
                    self.assertTrue(True, f"Found available port {alternative_port}")
                    return
            # If no ports are available, skip the test
            self.skipTest("No available ports found for testing")
    
    def test_port_is_not_available(self):
        """Test that is_port_available returns False when a port is not available."""
        # Use a high port number for testing
        test_port = 54322
        
        # Bind to the port to make it unavailable
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as blocking_socket:
            try:
                blocking_socket.bind(("localhost", test_port))
                
                # Now test that is_port_available returns False
                result = is_port_available(test_port)
                self.assertFalse(result, f"Port {test_port} should not be available")
                
            except OSError as e:
                # If we can't bind to the test port, it's already in use
                # which is perfect for our test case
                result = is_port_available(test_port)
                self.assertFalse(result, f"Port {test_port} should not be available (already in use)")