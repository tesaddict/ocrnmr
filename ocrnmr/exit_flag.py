"""Global exit flag for aggressive Ctrl-C handling."""

import threading

# Global force exit flag for aggressive Ctrl-C handling
FORCE_EXIT = threading.Event()

