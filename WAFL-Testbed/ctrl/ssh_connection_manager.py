"""
SSH Connection Manager for WAFL-Testbed.

This module provides a centralized manager for SSH connections to execution servers,
enabling connection reuse and automatic reconnection when sessions are dropped.
"""

import logging
import os
import threading
from typing import Dict, Optional

import paramiko


class SSHConnectionManager:
    """
    Manages SSH connections to execution servers with connection pooling and auto-reconnection.

    Features:
    - Connection pooling: Reuses existing SSH connections by IP address
    - Health checking: Detects dead connections before use
    - Auto-reconnection: Automatically reconnects when a session is dropped
    - Thread-safe: Uses locks for concurrent access
    """

    def __init__(
        self,
        user: str,
        private_key_path: str = "~/.ssh/id_ed25519",
        timeout: int = 10,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the SSH Connection Manager.

        Args:
            user: SSH username for connections
            private_key_path: Path to SSH private key (default: ~/.ssh/id_ed25519)
            timeout: Connection timeout in seconds (default: 10)
            logger: Optional logger instance
        """
        self.user = user
        self.private_key_path = os.path.expanduser(private_key_path)
        self.timeout = timeout
        self.logger = logger or logging.getLogger(__name__)

        if not os.path.exists(self.private_key_path):
            raise FileNotFoundError(f"SSH private key not found: {self.private_key_path}")

        self.key = paramiko.Ed25519Key.from_private_key_file(self.private_key_path)
        self._connections: Dict[str, paramiko.SSHClient] = {}
        self._lock = threading.Lock()

    def _log_debug(self, msg: str):
        """Log debug message."""
        if self.logger:
            self.logger.debug(msg)

    def _log_info(self, msg: str):
        """Log info message."""
        if self.logger:
            self.logger.info(msg)

    def _log_warning(self, msg: str):
        """Log warning message."""
        if self.logger:
            self.logger.warning(msg)

    def is_connection_alive(self, ssh: paramiko.SSHClient) -> bool:
        """
        Check if an SSH connection is still alive.

        Args:
            ssh: SSH client connection to check

        Returns:
            bool: True if connection is alive, False otherwise
        """
        if ssh is None:
            return False

        transport = ssh.get_transport()
        if transport is None:
            return False

        if not transport.is_active():
            return False

        # Send a keepalive to verify the connection is truly alive
        try:
            transport.send_ignore()
            return True
        except Exception:
            return False

    def _create_connection(self, ip: str) -> paramiko.SSHClient:
        """
        Create a new SSH connection.

        Args:
            ip: IP address to connect to

        Returns:
            paramiko.SSHClient: New SSH connection
        """
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(
            ip,
            port=22,
            username=self.user,
            pkey=self.key,
            timeout=self.timeout,
        )
        self._log_debug(f"🔗 Created new SSH connection to {ip}")
        return ssh

    def get_connection(self, ip: str) -> paramiko.SSHClient:
        """
        Get an SSH connection for the given IP address.

        Returns an existing connection if available and alive,
        otherwise creates a new connection. Automatically reconnects
        if the existing connection is dead.

        Args:
            ip: IP address to connect to

        Returns:
            paramiko.SSHClient: SSH connection (new or reused)

        Raises:
            paramiko.SSHException: If connection cannot be established
        """
        with self._lock:
            existing = self._connections.get(ip)

            if existing is not None:
                if self.is_connection_alive(existing):
                    self._log_debug(f"♻️ Reusing SSH connection to {ip}")
                    return existing
                else:
                    # Connection is dead, close it and create new one
                    self._log_warning(f"🔄 SSH connection to {ip} was dropped, reconnecting...")
                    try:
                        existing.close()
                    except Exception:
                        pass

            # Create new connection
            ssh = self._create_connection(ip)
            self._connections[ip] = ssh
            return ssh

    def close_connection(self, ip: str):
        """
        Close a specific SSH connection.

        Args:
            ip: IP address of the connection to close
        """
        with self._lock:
            if ip in self._connections:
                try:
                    self._connections[ip].close()
                    self._log_debug(f"🔌 Closed SSH connection to {ip}")
                except Exception as e:
                    self._log_warning(f"⚠️ Error closing SSH connection to {ip}: {e}")
                finally:
                    del self._connections[ip]

    def close_all(self):
        """
        Close all SSH connections.

        Should be called when the experiment is complete or on shutdown.
        """
        with self._lock:
            for ip, ssh in list(self._connections.items()):
                try:
                    ssh.close()
                    self._log_debug(f"🔌 Closed SSH connection to {ip}")
                except Exception as e:
                    self._log_warning(f"⚠️ Error closing SSH connection to {ip}: {e}")

            self._connections.clear()
            self._log_info("🔌 All SSH connections closed")

    def exec_command(
        self,
        ip: str,
        command: str,
        timeout: Optional[float] = None,
        get_pty: bool = False,
    ) -> tuple:
        """
        Execute a command on a remote host using a managed SSH connection.

        This is a convenience method that gets a connection and executes a command.
        If the connection is dead, it will automatically reconnect.

        Args:
            ip: IP address of the remote host
            command: Command to execute
            timeout: Command timeout (optional)
            get_pty: Whether to allocate a PTY (optional)

        Returns:
            tuple: (stdin, stdout, stderr) from exec_command
        """
        ssh = self.get_connection(ip)
        return ssh.exec_command(command, timeout=timeout, get_pty=get_pty)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close all connections."""
        self.close_all()
        return False

    @property
    def active_connections(self) -> int:
        """Return the number of active connections."""
        with self._lock:
            return len(self._connections)
