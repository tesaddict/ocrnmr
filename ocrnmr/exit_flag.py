"""Interrupt handling for aggressive Ctrl-C handling across threads.

This module provides a simple global flag that threads can check to see if
an interrupt (Ctrl-C) was requested. The signal handler sets this flag,
and threads check it periodically to exit cleanly.
"""

import threading

# Global force exit flag for aggressive Ctrl-C handling
# This is checked by background threads to exit early when Ctrl-C is pressed
FORCE_EXIT = threading.Event()
