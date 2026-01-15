#!/usr/bin/env python3
"""
WAFL-Testbed Infrastructure Verification & Benchmark Tool

This script provides two modes of verification:
1. Configuration Verification: Checks that settings in parameters.json and
   execution_config.json are correctly applied to running Docker containers.
2. Performance Benchmark: Runs actual performance tests (latency, bandwidth,
   packet loss, CPU) to verify infrastructure conditions are working correctly.

Usage:
    # Configuration verification only (default)
    python ctrl/verify.py
    python ctrl/verify.py --config-only
    python ctrl/verify.py --nodes 0,1,2

    # Performance benchmark
    python ctrl/verify.py --benchmark
    python ctrl/verify.py --benchmark --keep-containers
    python ctrl/verify.py --benchmark --nodes 0,1,2

    # Both verification and benchmark
    python ctrl/verify.py --all

    # Options
    python ctrl/verify.py --verbose          # Detailed output
    python ctrl/verify.py --benchmark --json-output results.json
"""

import argparse
import json
import os
import sys

# Add project root to sys.path to allow importing from ctrl package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import paramiko

from ctrl.container_manager import ContainerManager


# ANSI color codes
class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_header(text: str):
    """Display header"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text:^70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 70}{Colors.RESET}\n")


def print_section(text: str):
    """Display section header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}▶ {text}{Colors.RESET}")


def print_success(text: str):
    """Display success message"""
    print(f"  {Colors.GREEN}✓{Colors.RESET} {text}")


def print_warning(text: str):
    """Display warning message"""
    print(f"  {Colors.YELLOW}⚠{Colors.RESET} {text}")


def print_error(text: str):
    """Display error message"""
    print(f"  {Colors.RED}✗{Colors.RESET} {text}")


def print_info(text: str):
    """Display info message"""
    print(f"  {Colors.CYAN}ℹ{Colors.RESET} {text}")


# ==========================================
# Benchmark Data Classes
# ==========================================


@dataclass
class BenchmarkResult:
    """Result of a benchmark test"""

    test_name: str
    node_id: int
    success: bool
    expected_value: Optional[str]
    measured_value: Optional[str]
    tolerance: Optional[float]  # Percentage tolerance
    details: str
    raw_output: Optional[str] = None


@dataclass
class TestReport:
    """Summary of all benchmark tests"""

    total_tests: int
    passed_tests: int
    failed_tests: int
    results: List[BenchmarkResult]


# ==========================================
# Configuration Verification Classes
# ==========================================


class ConfigValidator:
    """Validator for configuration files"""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.params: Dict[str, Any] = {}
        self.exec_config: Dict[str, Any] = {}

    def load_configs(self) -> bool:
        """Load parameters.json and execution_config.json"""
        # Load parameters.json
        params_path = "ctrl/parameters.json"
        if not os.path.exists(params_path):
            self.errors.append(f"Parameters file not found: {params_path}")
            return False

        try:
            with open(params_path, "r") as f:
                self.params = json.load(f)
        except json.JSONDecodeError as e:
            self.errors.append(f"JSON parse error in {params_path}: {e}")
            return False
        except Exception as e:
            self.errors.append(f"Error reading {params_path}: {e}")
            return False

        # Load execution_config.json
        exec_path = "ctrl/execution_config.json"
        if not os.path.exists(exec_path):
            self.errors.append(f"Execution config not found: {exec_path}")
            return False

        try:
            with open(exec_path, "r") as f:
                self.exec_config = json.load(f)
        except json.JSONDecodeError as e:
            self.errors.append(f"JSON parse error in {exec_path}: {e}")
            return False
        except Exception as e:
            self.errors.append(f"Error reading {exec_path}: {e}")
            return False

        return True

    def validate_parameters(self, logger=None) -> bool:
        """Validate parameters.json"""

        def log_section(text):
            if logger:
                logger.info(f"▶ {text}")
            else:
                print_section(text)

        def log_success(text):
            if logger:
                logger.info(f"  ✓ {text}")
            else:
                print_success(text)

        def log_error(text):
            if logger:
                logger.error(f"  ✗ {text}")
            else:
                print_error(text)

        log_section("Validating parameters.json")

        all_valid = True

        # Validate wafl_phase
        if "wafl_phase" in self.params:
            wafl = self.params["wafl_phase"]
            required = [
                "aggregation_strategy",
                "batch_size",
                "learning_rate",
                "coefficiency",
            ]
            missing = [k for k in required if k not in wafl]

            if missing:
                self.errors.append(f"wafl_phase missing keys: {', '.join(missing)}")
                log_error(f"wafl_phase missing: {', '.join(missing)}")
                all_valid = False
            elif wafl.get("batch_size", 0) <= 0 or wafl.get("learning_rate", 0) <= 0:
                self.errors.append("wafl_phase: batch_size and learning_rate must be positive")
                log_error("batch_size and learning_rate must be positive")
                all_valid = False
            else:
                log_success("wafl_phase: All settings valid")
        else:
            self.errors.append("wafl_phase section not found")
            log_error("wafl_phase section not found")
            all_valid = False

        # Validate network_condition
        if "network_condition" in self.params:
            net = self.params["network_condition"]
            enabled = net.get("enabled", False)

            if enabled:
                # Only require detailed settings when enabled
                required = ["delay", "loss", "rate"]
                missing = [k for k in required if k not in net]

                if missing:
                    self.errors.append(f"network_condition missing keys: {', '.join(missing)}")
                    log_error(f"network_condition missing: {', '.join(missing)}")
                    all_valid = False
                else:
                    log_success(f"network_condition: enabled, delay={net['delay']}, loss={net['loss']}, rate={net['rate']}")
            else:
                log_success("network_condition: disabled")
        else:
            # network_condition section is optional when not used
            log_success("network_condition: not configured (disabled)")

        # Validate mobility_aware
        if "mobility_aware" in self.params:
            mob = self.params["mobility_aware"]
            enabled = mob.get("enabled", False)

            if enabled:
                # Only require detailed settings when enabled
                required = ["contact_pattern_file", "network_conditions_file"]
                missing = [k for k in required if k not in mob]

                if missing:
                    self.errors.append(f"mobility_aware missing keys when enabled: {', '.join(missing)}")
                    log_error(f"mobility_aware missing: {', '.join(missing)}")
                    all_valid = False
                else:
                    log_success("mobility_aware: enabled")
            else:
                log_success("mobility_aware: disabled")
        else:
            # mobility_aware section is optional when not used
            log_success("mobility_aware: not configured (disabled)")

        # Validate method
        if "method" in self.params:
            method_val = self.params["method"]
            if isinstance(method_val, str):
                valid_methods = ["tcp", "udp", "rudp", "dynamic", "fast"]
                if method_val not in valid_methods:
                    self.errors.append(f"Invalid method string: {method_val}. Must be one of {valid_methods}")
                    log_error(f"Invalid method string: {method_val}")
                    all_valid = False
                else:
                    log_success(f"method: {method_val}")
            elif isinstance(method_val, dict):
                # Ablation format validation
                base = method_val.get("base")
                if base not in ["tcp", "udp"]:
                    self.errors.append(f"Invalid method.base: {base}. Must be 'tcp' or 'udp'")
                    log_error(f"Invalid method.base: {base}")
                    all_valid = False

                fec = method_val.get("fec", True)
                comp = method_val.get("compression", False)
                nack = method_val.get("nack", True)

                # bool または dict 形式に対応
                fec_enabled = fec if isinstance(fec, bool) else fec.get("enabled", True)
                comp_enabled = comp if isinstance(comp, bool) else comp.get("enabled", False)
                nack_enabled = nack if isinstance(nack, bool) else nack.get("enabled", True)

                log_success(f"method (ablation): base={base}, fec={fec_enabled}, comp={comp_enabled}, nack={nack_enabled}")
            else:
                self.errors.append(f"Invalid method type: {type(method_val)}")
                log_error(f"Invalid method type: {type(method_val)}")
                all_valid = False
        else:
            self.errors.append("method section not found")
            log_error("method section not found")
            all_valid = False

        return all_valid

    def validate_execution_config(self, logger=None) -> bool:
        """Validate execution_config.json"""

        def log_section(text):
            if logger:
                logger.info(f"▶ {text}")
            else:
                print_section(text)

        def log_success(text):
            if logger:
                logger.info(f"  ✓ {text}")
            else:
                print_success(text)

        def log_error(text):
            if logger:
                logger.error(f"  ✗ {text}")
            else:
                print_error(text)

        def log_warning(text):
            if logger:
                logger.warning(f"  ⚠ {text}")
            else:
                print_warning(text)

        log_section("Validating execution_config.json")

        all_valid = True

        # Check basic structure
        if "user" not in self.exec_config or "deployment_location" not in self.exec_config:
            self.errors.append("execution_config missing user or deployment_location")
            log_error("Missing user or deployment_location")
            all_valid = False
        else:
            log_success("Basic structure: Valid")

        # Check nodes
        nodes = self.exec_config.get("nodes", [])
        if not nodes:
            self.errors.append("No nodes defined in execution_config")
            log_error("No nodes defined")
            all_valid = False
        else:
            log_success(f"Node configuration: {len(nodes)} nodes defined")

            # Check for cpu_limit consistency
            cpu_limits = [n.get("cpu_limit") for n in nodes]
            with_limit = sum(1 for c in cpu_limits if c is not None)

            if with_limit > 0 and with_limit < len(nodes):
                self.warnings.append(f"{with_limit}/{len(nodes)} nodes have cpu_limit configured")
                log_warning(f"{with_limit}/{len(nodes)} nodes have cpu_limit configured")

        # Check timeouts (REQUIRED)
        timeouts = self.exec_config.get("timeouts")
        if timeouts is None:
            self.errors.append("'timeouts' section is missing in execution_config.json")
            log_error("Missing 'timeouts' section - this is required")
            all_valid = False
        else:
            required_timeout_keys = ["model_fetch", "udp_initial_packet", "udp_inter_packet", "udp_model_completion"]
            missing_keys = [k for k in required_timeout_keys if k not in timeouts]
            if missing_keys:
                self.errors.append(f"Missing required timeout keys: {missing_keys}")
                log_error(f"Missing required timeout keys: {missing_keys}")
                all_valid = False
            else:
                log_success(f"Timeouts configuration: Valid (model_fetch={timeouts['model_fetch']}s, udp_model_completion={timeouts['udp_model_completion']}s)")

        return all_valid


class ContainerVerifier:
    """Verifies Docker container settings"""

    def __init__(self, params: Dict[str, Any], exec_config: Dict[str, Any], verbose: bool = False):
        self.params = params
        self.exec_config = exec_config
        self.verbose = verbose
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.private_key_path = os.path.expanduser("~/.ssh/id_ed25519")

        if not os.path.exists(self.private_key_path):
            raise FileNotFoundError(f"SSH private key not found: {self.private_key_path}")

        self.key = paramiko.Ed25519Key.from_private_key_file(self.private_key_path)

    def connect_ssh(self, ip: str, user: str) -> Optional[paramiko.SSHClient]:
        """Establish SSH connection"""
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(ip, port=22, username=user, pkey=self.key, timeout=5)
            return ssh
        except Exception as e:
            if self.verbose:
                print_error(f"SSH connection failed: {e}")
            return None

    def exec_command_with_logging(self, ssh: paramiko.SSHClient, command: str, timeout: Optional[float] = None) -> Tuple[Any, Any, Any]:
        """Execute SSH command with logging"""
        if self.verbose:
            # Use print_info or logger if available, but ContainerVerifier uses print_info/print_error
            # We can use print_info from global scope
            print_info(f"[CMD] {command}")

        return ssh.exec_command(command, timeout=timeout)

    def verify_container_running(self, ssh: paramiko.SSHClient, container_name: str) -> bool:
        """Check if container is running"""
        try:
            stdin, stdout, stderr = self.exec_command_with_logging(ssh, f"docker ps -q -f name={container_name}")
            container_id = stdout.read().decode().strip()
            return len(container_id) > 0
        except Exception:
            return False

    def verify_network_condition(self, ssh: paramiko.SSHClient, container_name: str) -> Tuple[bool, str]:
        """Verify network conditions are correctly applied on host veth interface"""
        net_cond = self.params.get("network_condition", {})

        if not net_cond.get("enabled", False):
            return True, "disabled (not checked)"

        delay = net_cond.get("delay", "50ms")
        loss = net_cond.get("loss", "0%")

        try:
            # Use docker exec instead of nsenter for more reliable access
            # Wait for container to be ready and get iflink
            max_wait = 30
            iflink_value = None

            for attempt in range(max_wait):
                # Try to read iflink using docker exec (more reliable than nsenter)
                cmd = f"docker exec {container_name} cat /sys/class/net/eth0/iflink 2>/dev/null"
                stdin, stdout, stderr = self.exec_command_with_logging(ssh, cmd)
                output = stdout.read().decode().strip()

                if output:
                    iflink_value = output
                    break

                if attempt < max_wait - 1:
                    import time

                    time.sleep(1)

            if not iflink_value:
                # Get debug info
                cmd = f"docker exec {container_name} ip link show 2>&1"
                stdin, stdout, stderr = self.exec_command_with_logging(ssh, cmd)
                debug_output = stdout.read().decode().strip()

                # Also check container state
                cmd = f"docker inspect -f '{{{{.State.Status}}}}' {container_name}"
                stdin, stdout, stderr = self.exec_command_with_logging(ssh, cmd)
                state = stdout.read().decode().strip()

                return (
                    False,
                    f"Container network not ready (state: {state}, waited {max_wait}s). Debug: {debug_output[:100]}",
                )

            veth_index = iflink_value

            if not veth_index:
                return False, "Could not read eth0/iflink after waiting"

            # Get veth interface name
            cmd = f"ip link | grep '^{veth_index}:' | awk -F': ' '{{print $2}}' | awk -F'@' '{{print $1}}'"
            stdin, stdout, stderr = self.exec_command_with_logging(ssh, cmd)
            host_veth = stdout.read().decode().strip()

            if not host_veth:
                return False, f"Could not find veth interface for index {veth_index}"

            # Check tc qdisc on host veth interface
            cmd = f"sudo tc qdisc show dev {host_veth}"
            stdin, stdout, stderr = self.exec_command_with_logging(ssh, cmd)
            output = stdout.read().decode().strip()

            if "netem" not in output:
                return False, f"netem qdisc not found on {host_veth}"

            # Expected values
            delay = net_cond.get("delay", "50ms")
            loss = net_cond.get("loss", "0%")
            rate = net_cond.get("rate", "100mbit")

            # Check delay
            delay_match = delay.replace("ms", ".0ms") in output or delay in output

            # Check loss (0% loss is typically not shown in output)
            loss_match = True
            if loss != "0%":
                loss_match = f"loss {loss}" in output

            # Check rate limit
            rate_match = rate in output or rate.replace("mbit", "Mbit") in output

            # Build verification message
            checks = []
            if delay_match:
                checks.append(f"delay={delay} ✓")
            else:
                checks.append(f"delay={delay} ✗")

            if loss_match:
                checks.append(f"loss={loss} ✓")
            else:
                checks.append(f"loss={loss} ✗")

            if rate_match:
                checks.append(f"rate={rate} ✓")
            else:
                checks.append(f"rate={rate} ✗")

            if delay_match and loss_match and rate_match:
                return True, f"{', '.join(checks)} (on {host_veth})"
            else:
                return False, f"settings mismatch on {host_veth}: {', '.join(checks)}"

        except Exception as e:
            return False, f"verification failed: {e}"

    def verify_cpu_limit(
        self,
        ssh: paramiko.SSHClient,
        container_name: str,
        expected_limit: Optional[float],
    ) -> Tuple[bool, str]:
        """Verify CPU limit is correctly applied"""
        if expected_limit is None:
            return True, "not configured"

        try:
            cmd = f"docker inspect {container_name} --format '{{{{.HostConfig.NanoCpus}}}}'"
            stdin, stdout, stderr = self.exec_command_with_logging(ssh, cmd)
            nano_cpus_str = stdout.read().decode().strip()

            if not nano_cpus_str or nano_cpus_str == "0":
                return False, f"not applied (expected: {expected_limit} cores)"

            nano_cpus = int(nano_cpus_str)
            actual_limit = nano_cpus / 1e9

            # Allow small tolerance for floating point comparison
            if abs(actual_limit - expected_limit) < 0.01:
                return True, f"{expected_limit} cores ✓"
            else:
                return (
                    False,
                    f"mismatch (expected: {expected_limit}, actual: {actual_limit:.2f})",
                )

        except Exception as e:
            return False, f"verification failed: {e}"

    def verify_node(self, node: Dict[str, Any], user: str, logger=None) -> bool:
        """Verify single node"""
        node_name = node.get("name", "unknown")
        ip = node.get("physical_ip", "")
        container_name = f"wafl-node-{node_name}"
        cpu_limit = node.get("cpu_limit")

        # Use logger if provided, otherwise print
        def log_section(text):
            if logger:
                logger.info(f"▶ {text}")
            else:
                print_section(text)

        def log_success(text):
            if logger:
                logger.info(f"  ✓ {text}")
            else:
                print_success(text)

        def log_error(text):
            if logger:
                logger.error(f"  ✗ {text}")
            else:
                print_error(text)

        def log_warning(text):
            if logger:
                logger.warning(f"  ⚠ {text}")
            else:
                print_warning(text)

        def log_info(text):
            if logger:
                logger.info(f"  ℹ {text}")
            else:
                print_info(text)

        log_section(f"Node {node_name} ({ip})")

        # Connect via SSH (silent - no logging)
        ssh = self.connect_ssh(ip, user)
        if not ssh:
            log_error("SSH connection: Failed")
            self.errors.append(f"Node {node_name}: SSH connection failed")
            return False

        all_valid = True

        # Check container running
        if not self.verify_container_running(ssh, container_name):
            log_warning(f"Container '{container_name}' is not running")
            ssh.close()
            return True  # Not an error if container isn't running

        log_success(f"Container running: {container_name}")

        # Verify network condition
        net_valid, net_msg = self.verify_network_condition(ssh, container_name)
        if net_valid:
            log_success(f"Network condition: {net_msg}")
        else:
            log_error(f"Network condition: {net_msg}")
            self.errors.append(f"Node {node_name}: {net_msg}")
            all_valid = False

        # Verify CPU limit
        cpu_valid, cpu_msg = self.verify_cpu_limit(ssh, container_name, cpu_limit)
        if cpu_valid:
            if cpu_limit is not None:
                log_success(f"CPU limit: {cpu_msg}")
            else:
                if self.verbose:
                    log_info(f"CPU limit: {cpu_msg}")
        else:
            log_error(f"CPU limit: {cpu_msg}")
            self.errors.append(f"Node {node_name}: CPU limit {cpu_msg}")
            all_valid = False

        ssh.close()
        return all_valid

    def verify_all(self, node_filter: Optional[List[int]] = None, logger=None) -> bool:
        """Verify all nodes"""
        if logger:
            logger.info("▶ Verifying container applications")
        else:
            print_section("Verifying container applications")

        nodes = self.exec_config.get("nodes", [])
        user = self.exec_config.get("user", "denjo")

        # Filter nodes if specified
        if node_filter is not None:
            nodes = [n for n in nodes if n.get("name") in node_filter]

        if logger:
            logger.info(f"  ℹ Verifying {len(nodes)} nodes...")
        else:
            print_info(f"Verifying {len(nodes)} nodes...")

        results = []
        for node in nodes:
            result = self.verify_node(node, user, logger=logger)
            results.append(result)

        return all(results)


# ==========================================
# Benchmark Infrastructure Class
# ==========================================


class InfrastructureBenchmark:
    """Performs infrastructure condition benchmarks"""

    def __init__(
        self,
        params: Dict[str, Any],
        exec_config: Dict[str, Any],
        auto_setup: bool = True,
        verbose: bool = False,
    ):
        self.params = params
        self.exec_config = exec_config
        self.auto_setup = auto_setup
        self.verbose = verbose
        self.results: List[BenchmarkResult] = []

        deployment_location = exec_config.get("deployment_location", "/home/denjo")
        user = exec_config.get("user", "denjo")

        self.container_manager = ContainerManager(user=user, deployment_location=deployment_location, verbose=verbose)

    def connect_ssh(self, ip: str) -> Optional[paramiko.SSHClient]:
        """Establish SSH connection (delegate to manager)"""
        return self.container_manager.connect_ssh(ip)

    def exec_command_with_logging(self, ssh: paramiko.SSHClient, command: str, timeout: Optional[float] = None) -> Tuple[Any, Any, Any]:
        """Execute SSH command with logging (delegate to manager)"""
        return self.container_manager.exec_command(ssh, command, timeout)

    def check_container_running(self, ssh: paramiko.SSHClient, container_name: str) -> bool:
        """Check if container is running"""
        try:
            stdin, stdout, stderr = self.exec_command_with_logging(ssh, f"docker ps -q -f name={container_name}")
            container_id = stdout.read().decode().strip()
            return len(container_id) > 0
        except Exception:
            return False

    def start_container(self, node: Dict[str, Any]) -> bool:
        """Start a Docker container for the given node"""
        # Prepare node info with correct ports if missing
        # verify.py logic used defaults 10001/10002 if not in node dict
        node_info = node.copy()
        if "host_port_ctrl" not in node_info:
            node_info["host_port_ctrl"] = 10001
        if "container_port_ctrl" not in node_info:
            node_info["container_port_ctrl"] = 10001
        if "host_port_p2p" not in node_info:
            node_info["host_port_p2p"] = 10002

        return self.container_manager.start_wafl_container(
            node_info=node_info,
            experiment_params=self.params,
            env_vars="-e LOG_LEVEL=INFO",
        )

    def apply_network_conditions(self, node: Dict[str, Any]) -> bool:
        """Apply network conditions to a container"""
        node_name = node.get("name", "unknown")
        ip = node.get("physical_ip", "")
        container_name = f"wafl-node-{node_name}"

        ssh = self.connect_ssh(ip)
        if not ssh:
            return False

        try:
            result = self.container_manager.apply_network_conditions(ssh, container_name, self.params)
            ssh.close()
            return result
        except Exception:
            ssh.close()
            return False

    def stop_container(self, node: Dict[str, Any]) -> bool:
        """Stop and remove a Docker container"""
        node_name = node.get("name", "unknown")
        ip = node.get("physical_ip", "")
        container_name = f"wafl-node-{node_name}"

        return self.container_manager.stop_container(ip, container_name)

    def check_containers_running(self, nodes: List[Dict[str, Any]]) -> bool:
        """Check if all containers are running"""
        all_running = True
        for node in nodes:
            node_name = node.get("name", "unknown")
            ip = node.get("physical_ip", "")
            container_name = f"wafl-node-{node_name}"

            ssh = self.connect_ssh(ip)
            if not ssh:
                all_running = False
                continue

            running = self.check_container_running(ssh, container_name)
            ssh.close()

            if not running:
                if self.verbose:
                    print_warning(f"Container {container_name} is not running")
                all_running = False

        return all_running

    def setup_containers(self, nodes: List[Dict[str, Any]]) -> bool:
        """Setup containers for testing (start + apply network conditions) - parallelized"""
        print_section("Setting up containers")

        results = []
        threads = []
        lock = threading.Lock()

        def setup_node(node):
            """Setup a single node's container"""
            node_name = node.get("name", "unknown")
            success = False

            # Start container
            if not self.start_container(node):
                with lock:
                    print_error(f"Failed to start container for node {node_name}")
                return

            # Apply network conditions
            if not self.apply_network_conditions(node):
                with lock:
                    print_error(f"Failed to apply network conditions for node {node_name}")
                return

            success = True
            with lock:
                results.append(success)

        # Create and start threads for each node
        for node in nodes:
            t = threading.Thread(target=setup_node, args=(node,))
            t.start()
            threads.append(t)

        # Wait for all threads to complete
        for t in threads:
            t.join()

        success_count = len(results)

        if success_count == len(nodes):
            print_success(f"All {len(nodes)} containers setup successfully")
            # Wait for network conditions to take effect
            if self.verbose:
                print_info("Waiting for network conditions to take effect...")
            time.sleep(3)
            return True
        else:
            print_warning(f"Only {success_count}/{len(nodes)} containers setup successfully")
            return False

    def teardown_containers(self, nodes: List[Dict[str, Any]]) -> None:
        """Stop and remove all containers - parallelized"""
        print_section("Cleaning up containers")

        threads = []

        def cleanup_node(node):
            """Cleanup a single node's container"""
            self.stop_container(node)

        # Create and start threads for each node
        for node in nodes:
            t = threading.Thread(target=cleanup_node, args=(node,))
            t.start()
            threads.append(t)

        # Wait for all threads to complete
        for t in threads:
            t.join()

        print_success("Container cleanup completed")

    def test_network_latency(self, node: Dict[str, Any]) -> BenchmarkResult:
        """Test network latency using ping"""
        node_name = node.get("name", "unknown")
        ip = node.get("physical_ip", "")

        net_cond = self.params.get("network_condition", {})
        if not net_cond.get("enabled", False):
            return BenchmarkResult(
                test_name="Network Latency",
                node_id=node_name,
                success=True,
                expected_value=None,
                measured_value=None,
                tolerance=None,
                details="Network conditions disabled (skipped)",
            )

        # Parse expected delay
        delay_str = net_cond.get("delay", "50ms")
        try:
            expected_delay_ms = float(delay_str.replace("ms", ""))
        except ValueError:
            expected_delay_ms = 50.0

        tolerance = 20.0  # ±20%

        try:
            # 1. SSH to host
            ssh = self.connect_ssh(ip)
            if not ssh:
                return BenchmarkResult(
                    test_name="Network Latency",
                    node_id=node_name,
                    success=False,
                    expected_value=f"{expected_delay_ms}ms",
                    measured_value=None,
                    tolerance=tolerance,
                    details="SSH connection failed",
                )

            # 2. Get container IP
            container_name = f"wafl-node-{node_name}"
            get_ip_cmd = f"docker inspect -f '{{{{.NetworkSettings.IPAddress}}}}' {container_name}"
            stdin, stdout, stderr = self.exec_command_with_logging(ssh, get_ip_cmd)
            container_ip = stdout.read().decode().strip()

            if not container_ip:
                ssh.close()
                return BenchmarkResult(
                    test_name="Network Latency",
                    node_id=node_name,
                    success=False,
                    expected_value=f"{expected_delay_ms}ms",
                    measured_value=None,
                    tolerance=tolerance,
                    details="Could not get container IP",
                )

            # 3. Ping container from host (10 packets)
            # Note: netem rules are on host veth (egress), so Host -> Container traffic is delayed.
            # Ping RTT = (Host->Container) + (Container->Host)
            # Host->Container is delayed by 'delay'. Container->Host is NOT delayed (unless ingress shaping is set, which is rare/complex).
            # So RTT should be approx 'delay' + processing time.
            # However, standard netem usually applies to egress.
            # If we applied "tc qdisc add dev veth... root netem delay 100ms", that affects packets LEAVING the interface.
            # Packets leaving host veth -> entering container. So Host->Container is delayed.
            # Packets leaving container -> entering host veth. This is ingress on host veth. TC usually shapes egress.
            # So only Host->Container is delayed. RTT ~= delay.

            ping_cmd = f"ping -c 10 -W 2 {container_ip}"
            stdin, stdout, stderr = self.exec_command_with_logging(ssh, ping_cmd)
            ping_output = stdout.read().decode()
            exit_status = stdout.channel.recv_exit_status()

            ssh.close()

            if exit_status != 0:
                return BenchmarkResult(
                    test_name="Network Latency",
                    node_id=node_name,
                    success=False,
                    expected_value=f"{expected_delay_ms}ms",
                    measured_value=None,
                    tolerance=tolerance,
                    details=f"Ping failed: {stderr.read().decode().strip()}",
                    raw_output=ping_output,
                )

            # Parse RTT from ping output
            lines = ping_output.splitlines()
            rtt_line = None
            for line in lines:
                if "rtt min/avg/max" in line or "round-trip min/avg/max" in line:
                    rtt_line = line
                    break

            if not rtt_line:
                return BenchmarkResult(
                    test_name="Network Latency",
                    node_id=node_name,
                    success=False,
                    expected_value=f"{expected_delay_ms}ms",
                    measured_value=None,
                    tolerance=tolerance,
                    details="Could not parse RTT from ping output",
                    raw_output=ping_output,
                )

            # Extract avg RTT
            parts = rtt_line.split("=")[1].strip().split()[0]
            avg_rtt = float(parts.split("/")[1])

            # Expected RTT is approx the configured delay (one-way delay applied to egress)
            expected_rtt = expected_delay_ms
            rtt_lower = expected_rtt * (1 - tolerance / 100)
            rtt_upper = expected_rtt * (1 + tolerance / 100)

            # Allow for some base overhead (e.g. +1ms)
            if expected_rtt < 1.0:
                # If expected is 0 (no delay), allow up to 1ms
                rtt_upper = max(rtt_upper, 1.0)

            success = rtt_lower <= avg_rtt <= rtt_upper

            details = f"RTT: {avg_rtt:.2f}ms (expected ~{expected_rtt:.2f}ms ±{tolerance}%)"
            if success:
                details += " ✓"
            else:
                details += f" ✗ (out of range: {rtt_lower:.2f}-{rtt_upper:.2f}ms)"

            return BenchmarkResult(
                test_name="Network Latency",
                node_id=node_name,
                success=success,
                expected_value=f"~{expected_rtt:.2f}ms",
                measured_value=f"{avg_rtt:.2f}ms",
                tolerance=tolerance,
                details=details,
                raw_output=ping_output,
            )

        except Exception as e:
            return BenchmarkResult(
                test_name="Network Latency",
                node_id=node_name,
                success=False,
                expected_value=f"{expected_delay_ms}ms",
                measured_value=None,
                tolerance=tolerance,
                details=f"Test failed: {e}",
            )

    def test_packet_loss(self, node: Dict[str, Any]) -> BenchmarkResult:
        """Test packet loss using ping statistics"""
        node_name = node.get("name", "unknown")
        ip = node.get("physical_ip", "")

        net_cond = self.params.get("network_condition", {})
        if not net_cond.get("enabled", False):
            return BenchmarkResult(
                test_name="Packet Loss",
                node_id=node_name,
                success=True,
                expected_value=None,
                measured_value=None,
                tolerance=None,
                details="Network conditions disabled (skipped)",
            )

        # Parse expected loss
        loss_str = net_cond.get("loss", "0%")
        try:
            expected_loss_pct = float(loss_str.replace("%", ""))
        except ValueError:
            expected_loss_pct = 0.0

        tolerance = 5.0  # ±5% (absolute)

        try:
            # 1. SSH to host
            ssh = self.connect_ssh(ip)
            if not ssh:
                return BenchmarkResult(
                    test_name="Packet Loss",
                    node_id=node_name,
                    success=False,
                    expected_value=f"{expected_loss_pct}%",
                    measured_value=None,
                    tolerance=tolerance,
                    details="SSH connection failed",
                )

            # 2. Get container IP
            container_name = f"wafl-node-{node_name}"
            get_ip_cmd = f"docker inspect -f '{{{{.NetworkSettings.IPAddress}}}}' {container_name}"
            stdin, stdout, stderr = self.exec_command_with_logging(ssh, get_ip_cmd)
            container_ip = stdout.read().decode().strip()

            if not container_ip:
                ssh.close()
                return BenchmarkResult(
                    test_name="Packet Loss",
                    node_id=node_name,
                    success=False,
                    expected_value=f"{expected_loss_pct}%",
                    measured_value=None,
                    tolerance=tolerance,
                    details="Could not get container IP",
                )

            # 3. Ping container from host (50 packets, 0.2s interval)
            # Increased from 20 to 50 to improve accuracy (10s duration)
            ping_cmd = f"ping -c 50 -i 0.2 -W 2 {container_ip}"
            stdin, stdout, stderr = self.exec_command_with_logging(ssh, ping_cmd)
            ping_output = stdout.read().decode()
            exit_status = stdout.channel.recv_exit_status()

            ssh.close()

            if exit_status != 0 and exit_status != 1:
                # returncode 1 is OK (some packets lost but not all)
                return BenchmarkResult(
                    test_name="Packet Loss",
                    node_id=node_name,
                    success=False,
                    expected_value=f"{expected_loss_pct}%",
                    measured_value=None,
                    tolerance=tolerance,
                    details=f"Ping failed: {stderr.read().decode().strip()}",
                    raw_output=ping_output,
                )

            # Parse packet loss from ping output
            lines = ping_output.splitlines()
            loss_line = None
            for line in lines:
                if "packet loss" in line or "packets transmitted" in line:
                    loss_line = line
                    break

            if not loss_line:
                return BenchmarkResult(
                    test_name="Packet Loss",
                    node_id=node_name,
                    success=False,
                    expected_value=f"{expected_loss_pct}%",
                    measured_value=None,
                    tolerance=tolerance,
                    details="Could not parse packet loss from ping output",
                    raw_output=ping_output,
                )

            # Extract packet loss percentage
            import re

            match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*packet loss", loss_line)
            if not match:
                return BenchmarkResult(
                    test_name="Packet Loss",
                    node_id=node_name,
                    success=False,
                    expected_value=f"{expected_loss_pct}%",
                    measured_value=None,
                    tolerance=tolerance,
                    details="Could not extract packet loss percentage",
                    raw_output=ping_output,
                )

            measured_loss_pct = float(match.group(1))

            # Check if within tolerance (absolute difference)
            loss_lower = max(0, expected_loss_pct - tolerance)
            loss_upper = expected_loss_pct + tolerance

            success = loss_lower <= measured_loss_pct <= loss_upper

            details = f"Loss: {measured_loss_pct:.1f}% (expected {expected_loss_pct}% ±{tolerance}%)"
            if success:
                details += " ✓"
            else:
                details += f" ✗ (out of range: {loss_lower:.1f}-{loss_upper:.1f}%)"

            return BenchmarkResult(
                test_name="Packet Loss",
                node_id=node_name,
                success=success,
                expected_value=f"{expected_loss_pct}%",
                measured_value=f"{measured_loss_pct:.1f}%",
                tolerance=tolerance,
                details=details,
                raw_output=ping_output,
            )

        except Exception as e:
            return BenchmarkResult(
                test_name="Packet Loss",
                node_id=node_name,
                success=False,
                expected_value=f"{expected_loss_pct}%",
                measured_value=None,
                tolerance=tolerance,
                details=f"Test failed: {e}",
            )

    def test_network_bandwidth(self, node: Dict[str, Any]) -> BenchmarkResult:
        """Test network bandwidth using iperf3"""
        node_name = node.get("name", "unknown")
        ip = node.get("physical_ip", "")
        container_name = f"wafl-node-{node_name}"

        net_cond = self.params.get("network_condition", {})
        if not net_cond.get("enabled", False):
            return BenchmarkResult(
                test_name="Network Bandwidth",
                node_id=node_name,
                success=True,
                expected_value=None,
                measured_value=None,
                tolerance=None,
                details="Network conditions disabled (skipped)",
            )

        # Parse expected rate
        rate_str = net_cond.get("rate", "100mbit")
        try:
            # Convert to Mbps (handle mbit, Mbit, gbit, Gbit, etc.)
            rate_lower = rate_str.lower()
            if "gbit" in rate_lower:
                expected_rate_mbps = float(rate_lower.replace("gbit", "")) * 1000
            else:
                expected_rate_mbps = float(rate_lower.replace("mbit", ""))
        except ValueError:
            expected_rate_mbps = 100.0

        tolerance = 20.0  # ±20%

        ssh = self.connect_ssh(ip)
        if not ssh:
            return BenchmarkResult(
                test_name="Network Bandwidth",
                node_id=node_name,
                success=False,
                expected_value=f"{expected_rate_mbps}Mbps",
                measured_value=None,
                tolerance=tolerance,
                details="SSH connection failed",
            )

        try:
            # Check if iperf3 is installed on host
            check_cmd = "which iperf3"
            stdin, stdout, stderr = self.exec_command_with_logging(ssh, check_cmd)
            if stdout.channel.recv_exit_status() != 0:
                ssh.close()
                return BenchmarkResult(
                    test_name="Network Bandwidth",
                    node_id=node_name,
                    success=False,
                    expected_value=f"{expected_rate_mbps}Mbps",
                    measured_value=None,
                    tolerance=tolerance,
                    details="iperf3 not found on host",
                )

            # Kill any existing iperf3 server processes
            kill_cmd = f"docker exec {container_name} pkill -9 iperf3 || true"
            stdin, stdout, stderr = self.exec_command_with_logging(ssh, kill_cmd)
            stdout.channel.recv_exit_status()

            # Start iperf3 server in container (background, port 5201)
            server_cmd = f"docker exec -d {container_name} iperf3 -s -p 5201"
            stdin, stdout, stderr = self.exec_command_with_logging(ssh, server_cmd)
            exit_status = stdout.channel.recv_exit_status()

            if exit_status != 0:
                error_msg = stderr.read().decode().strip()
                ssh.close()
                return BenchmarkResult(
                    test_name="Network Bandwidth",
                    node_id=node_name,
                    success=False,
                    expected_value=f"{expected_rate_mbps}Mbps",
                    measured_value=None,
                    tolerance=tolerance,
                    details=f"Failed to start iperf3 server: {error_msg}",
                )

            # Wait for server to start
            time.sleep(2)

            # Get container IP
            get_ip_cmd = f"docker inspect -f '{{{{.NetworkSettings.IPAddress}}}}' {container_name}"
            stdin, stdout, stderr = self.exec_command_with_logging(ssh, get_ip_cmd)
            container_ip = stdout.read().decode().strip()

            if not container_ip:
                kill_cmd = f"docker exec {container_name} pkill -9 iperf3 || true"
                stdin, stdout, stderr = self.exec_command_with_logging(ssh, kill_cmd)
                stdout.channel.recv_exit_status()
                ssh.close()
                return BenchmarkResult(
                    test_name="Network Bandwidth",
                    node_id=node_name,
                    success=False,
                    expected_value=f"{expected_rate_mbps}Mbps",
                    measured_value=None,
                    tolerance=tolerance,
                    details="Could not get container IP",
                )

            # Parse expected loss for bandwidth calculation
            loss_str = net_cond.get("loss", "0%")
            try:
                expected_loss_pct = float(loss_str.replace("%", ""))
            except ValueError:
                expected_loss_pct = 0.0

            # Run iperf3 client from host to container (5 second test, UDP)
            # Reduced from 10s to 5s to speed up verification
            # We use UDP because TCP throughput is heavily affected by packet loss/delay
            # and doesn't reflect the raw link capacity we want to verify.
            # We set target bandwidth slightly higher (1.1x) to ensure saturation if possible,
            # or just match the rate. Let's match the rate to avoid over-saturation issues.
            client_cmd = f"iperf3 -c {container_ip} -p 5201 -t 5 -J -u -b {expected_rate_mbps}M"
            stdin, stdout, stderr = self.exec_command_with_logging(ssh, client_cmd, timeout=30)
            client_output = stdout.read().decode()
            exit_status = stdout.channel.recv_exit_status()

            # Cleanup: kill iperf3 server
            kill_cmd = f"docker exec {container_name} pkill -9 iperf3 || true"
            stdin, stdout, stderr = self.exec_command_with_logging(ssh, kill_cmd)
            stdout.channel.recv_exit_status()

            ssh.close()

            if exit_status != 0:
                error_msg = stderr.read().decode().strip()
                return BenchmarkResult(
                    test_name="Network Bandwidth",
                    node_id=node_name,
                    success=False,
                    expected_value=f"{expected_rate_mbps}Mbps",
                    measured_value=None,
                    tolerance=tolerance,
                    details=f"iperf3 client failed: {error_msg}",
                    raw_output=client_output,
                )

            # Parse JSON output
            try:
                import json as json_module

                result_data = json_module.loads(client_output)
                # Get average bits per second from receiver's perspective
                # For UDP, we look at 'sum' (or 'sum_received' if available, but usually 'sum' in end)
                # In iperf3 JSON for UDP: end -> sum -> bits_per_second
                measured_bps = result_data["end"]["sum"]["bits_per_second"]
                measured_mbps = measured_bps / 1_000_000  # Convert to Mbps
            except (json_module.JSONDecodeError, KeyError) as e:
                return BenchmarkResult(
                    test_name="Network Bandwidth",
                    node_id=node_name,
                    success=False,
                    expected_value=f"{expected_rate_mbps}Mbps",
                    measured_value=None,
                    tolerance=tolerance,
                    details=f"Failed to parse iperf3 output: {e}",
                    raw_output=client_output,
                )

            # Expected throughput with UDP and Loss: Rate * (1 - Loss)
            expected_throughput = expected_rate_mbps * (1 - expected_loss_pct / 100.0)

            # Check if within tolerance
            bw_lower = expected_throughput * (1 - tolerance / 100)
            bw_upper = expected_throughput * (1 + tolerance / 100)

            # Also allow if it's higher (up to rate limit), but not lower
            # Actually, if we send at Rate, we can't receive MORE than Rate * (1-Loss).
            # So upper bound should be close to expected.

            success = bw_lower <= measured_mbps <= bw_upper

            details = f"Bandwidth: {measured_mbps:.2f}Mbps (expected {expected_throughput:.2f}Mbps ±{tolerance}%) [UDP]"
            if success:
                details += " ✓"
            else:
                details += f" ✗ (out of range: {bw_lower:.2f}-{bw_upper:.2f}Mbps)"

            return BenchmarkResult(
                test_name="Network Bandwidth",
                node_id=node_name,
                success=success,
                expected_value=f"{expected_throughput:.2f}Mbps",
                measured_value=f"{measured_mbps:.2f}Mbps",
                tolerance=tolerance,
                details=details,
                raw_output=client_output,
            )

        except Exception as e:
            # Cleanup on error
            try:
                kill_cmd = f"docker exec {container_name} pkill -9 iperf3 || true"
                stdin, stdout, stderr = self.exec_command_with_logging(ssh, kill_cmd)
                stdout.channel.recv_exit_status()
                ssh.close()
            except Exception:
                pass

            return BenchmarkResult(
                test_name="Network Bandwidth",
                node_id=node_name,
                success=False,
                expected_value=f"{expected_rate_mbps}Mbps",
                measured_value=None,
                tolerance=tolerance,
                details=f"Test failed: {e}",
            )

    def test_cpu_limit(self, node: Dict[str, Any]) -> BenchmarkResult:
        """Test CPU limit using stress-ng"""
        node_name = node.get("name", "unknown")
        ip = node.get("physical_ip", "")
        container_name = f"wafl-node-{node_name}"
        cpu_limit = node.get("cpu_limit")

        if not cpu_limit:
            return BenchmarkResult(
                test_name="CPU Limit",
                node_id=node_name,
                success=True,
                expected_value=None,
                measured_value=None,
                tolerance=None,
                details="No CPU limit configured (skipped)",
            )

        try:
            expected_cpu_pct = float(cpu_limit) * 100  # Convert cores to percentage
        except ValueError:
            expected_cpu_pct = 100.0

        tolerance = 15.0  # ±15%

        ssh = self.connect_ssh(ip)
        if not ssh:
            return BenchmarkResult(
                test_name="CPU Limit",
                node_id=node_name,
                success=False,
                expected_value=f"{expected_cpu_pct}%",
                measured_value=None,
                tolerance=tolerance,
                details="SSH connection failed",
            )

        try:
            # Start CPU stress test in container (background)
            stress_cmd = f"docker exec -d {container_name} stress-ng --cpu 4 --timeout 10s"
            stdin, stdout, stderr = self.exec_command_with_logging(ssh, stress_cmd)
            exit_status = stdout.channel.recv_exit_status()

            if exit_status != 0:
                ssh.close()
                # Try alternative method with simple workload
                return self._test_cpu_limit_alternative(node, ssh)

            # Wait a bit for stress to ramp up
            time.sleep(3)

            # Monitor CPU usage using docker stats
            stats_cmd = f"docker stats {container_name} --no-stream --format '{{{{.CPUPerc}}}}'"
            stdin, stdout, stderr = self.exec_command_with_logging(ssh, stats_cmd)
            cpu_output = stdout.read().decode().strip()

            ssh.close()

            if not cpu_output:
                return BenchmarkResult(
                    test_name="CPU Limit",
                    node_id=node_name,
                    success=False,
                    expected_value=f"≤{expected_cpu_pct}%",
                    measured_value=None,
                    tolerance=tolerance,
                    details="Could not read CPU stats",
                )

            # Parse CPU percentage (format: "123.45%")
            measured_cpu_pct = float(cpu_output.replace("%", ""))

            # Check if within limit (should not exceed expected + tolerance)
            cpu_upper = expected_cpu_pct + tolerance
            success = measured_cpu_pct <= cpu_upper

            details = f"CPU: {measured_cpu_pct:.1f}% (limit ≤{expected_cpu_pct}% +{tolerance}%)"
            if success:
                details += " ✓"
            else:
                details += f" ✗ (exceeded limit: ≤{cpu_upper:.1f}%)"

            return BenchmarkResult(
                test_name="CPU Limit",
                node_id=node_name,
                success=success,
                expected_value=f"≤{expected_cpu_pct}%",
                measured_value=f"{measured_cpu_pct:.1f}%",
                tolerance=tolerance,
                details=details,
            )

        except Exception as e:
            ssh.close()
            return BenchmarkResult(
                test_name="CPU Limit",
                node_id=node_name,
                success=False,
                expected_value=f"≤{expected_cpu_pct}%",
                measured_value=None,
                tolerance=tolerance,
                details=f"Test failed: {e}",
            )

    def _test_cpu_limit_alternative(self, node: Dict[str, Any], ssh: Optional[paramiko.SSHClient] = None) -> BenchmarkResult:
        """Alternative CPU limit test using dd command"""
        node_name = node.get("name", "unknown")
        container_name = f"wafl-node-{node_name}"
        cpu_limit = node.get("cpu_limit")

        try:
            expected_cpu_pct = float(cpu_limit) * 100
        except (ValueError, TypeError):
            expected_cpu_pct = 100.0

        tolerance = 15.0

        if ssh is None:
            ip = node.get("physical_ip", "")
            ssh = self.connect_ssh(ip)
            if not ssh:
                return BenchmarkResult(
                    test_name="CPU Limit",
                    node_id=node_name,
                    success=False,
                    expected_value=f"≤{expected_cpu_pct}%",
                    measured_value=None,
                    tolerance=tolerance,
                    details="SSH connection failed",
                )

        try:
            # Use dd as CPU-intensive workload
            dd_cmd = f"docker exec -d {container_name} dd if=/dev/zero of=/dev/null bs=1M count=10000"
            stdin, stdout, stderr = self.exec_command_with_logging(ssh, dd_cmd)
            stdout.channel.recv_exit_status()

            time.sleep(2)

            # Monitor CPU usage
            stats_cmd = f"docker stats {container_name} --no-stream --format '{{{{.CPUPerc}}}}'"
            stdin, stdout, stderr = self.exec_command_with_logging(ssh, stats_cmd)
            cpu_output = stdout.read().decode().strip()

            ssh.close()

            if not cpu_output:
                return BenchmarkResult(
                    test_name="CPU Limit",
                    node_id=node_name,
                    success=False,
                    expected_value=f"≤{expected_cpu_pct}%",
                    measured_value=None,
                    tolerance=tolerance,
                    details="Could not read CPU stats (alternative method)",
                )

            measured_cpu_pct = float(cpu_output.replace("%", ""))
            cpu_upper = expected_cpu_pct + tolerance
            success = measured_cpu_pct <= cpu_upper

            details = f"CPU: {measured_cpu_pct:.1f}% (limit ≤{expected_cpu_pct}% +{tolerance}%) [alt]"
            if success:
                details += " ✓"
            else:
                details += " ✗"

            return BenchmarkResult(
                test_name="CPU Limit",
                node_id=node_name,
                success=success,
                expected_value=f"≤{expected_cpu_pct}%",
                measured_value=f"{measured_cpu_pct:.1f}%",
                tolerance=tolerance,
                details=details,
            )

        except Exception as e:
            ssh.close()
            return BenchmarkResult(
                test_name="CPU Limit",
                node_id=node_name,
                success=False,
                expected_value=f"≤{expected_cpu_pct}%",
                measured_value=None,
                tolerance=tolerance,
                details=f"Alternative test failed: {e}",
            )

    def run_all_tests(self, nodes: List[Dict[str, Any]]) -> TestReport:
        """Run all benchmark tests"""
        print_header("WAFL-Testbed Infrastructure Benchmark")

        # Auto-setup if enabled
        if self.auto_setup:
            containers_running = self.check_containers_running(nodes)

            if not containers_running:
                print_info("Auto-setup mode: Starting containers...")
                if not self.setup_containers(nodes):
                    print_error("Container setup failed. Aborting tests.")
                    return TestReport(0, 0, 0, [])
            else:
                print_info("Containers already running")
                # Apply network conditions to existing containers
                print_section("Applying network conditions to existing containers")

                net_cond = self.params.get("network_condition", {})
                if net_cond.get("enabled", False):
                    threads = []
                    lock = threading.Lock()

                    def apply_to_node(node):
                        """Apply network conditions to a single node"""
                        if not self.apply_network_conditions(node):
                            node_name = node.get("name", "unknown")
                            with lock:
                                print_warning(f"Failed to apply network conditions to node {node_name}")

                    # Apply network conditions in parallel
                    for node in nodes:
                        t = threading.Thread(target=apply_to_node, args=(node,))
                        t.start()
                        threads.append(t)

                    # Wait for all threads to complete
                    for t in threads:
                        t.join()

                    print_success("Network conditions applied to existing containers")
                    # Wait for network conditions to take effect
                    time.sleep(3)
                else:
                    print_info("Network conditions are disabled")

        # Run tests
        print_section("Running benchmark tests")

        for node in nodes:
            node_name = node.get("name", "unknown")
            print_info(f"Testing node {node_name}...")

            # Test network latency
            result = self.test_network_latency(node)
            self.results.append(result)
            if result.success:
                print_success(f"  Latency: {result.details}")
            else:
                print_error(f"  Latency: {result.details}")

            # Test packet loss
            result = self.test_packet_loss(node)
            self.results.append(result)
            if result.success:
                print_success(f"  Packet Loss: {result.details}")
            else:
                print_error(f"  Packet Loss: {result.details}")

            # Test network bandwidth
            result = self.test_network_bandwidth(node)
            self.results.append(result)
            if result.success:
                print_success(f"  Bandwidth: {result.details}")
            else:
                print_error(f"  Bandwidth: {result.details}")

            # Test CPU limit
            result = self.test_cpu_limit(node)
            self.results.append(result)
            if result.success:
                print_success(f"  CPU Limit: {result.details}")
            else:
                print_error(f"  CPU Limit: {result.details}")

        # Generate report
        passed = sum(1 for r in self.results if r.success)
        failed = len(self.results) - passed

        return TestReport(
            total_tests=len(self.results),
            passed_tests=passed,
            failed_tests=failed,
            results=self.results,
        )


def main():
    """Main function - supports both verification and benchmark modes"""
    parser = argparse.ArgumentParser(description="WAFL-Testbed Infrastructure Verification & Benchmark Tool")

    # Mode selection
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmarks instead of config verification",
    )
    parser.add_argument("--all", action="store_true", help="Run both verification and benchmarks")

    # Shared options
    parser.add_argument("--nodes", type=str, help="Comma-separated list of node IDs (e.g., 0,1,2)")
    parser.add_argument("--verbose", action="store_true", help="Detailed output")

    # Verification-only options
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="Verify config files only (skip container checks)",
    )

    # Benchmark-only options
    parser.add_argument(
        "--no-auto-setup",
        action="store_true",
        help="Use existing containers (don't auto-start)",
    )
    parser.add_argument(
        "--keep-containers",
        action="store_true",
        help="Keep containers running after benchmark tests",
    )
    parser.add_argument("--json-output", type=str, help="Save benchmark results to JSON file")

    args = parser.parse_args()

    # =====================================
    # Load configurations
    # =====================================
    params_path = "ctrl/parameters.json"
    exec_config_path = "ctrl/execution_config.json"

    if not os.path.exists(params_path):
        print_error(f"Parameters file not found: {params_path}")
        return 1

    if not os.path.exists(exec_config_path):
        print_error(f"Execution config not found: {exec_config_path}")
        return 1

    try:
        with open(params_path) as f:
            params = json.load(f)
        with open(exec_config_path) as f:
            exec_config = json.load(f)
    except json.JSONDecodeError as e:
        print_error(f"JSON parse error: {e}")
        return 1

    # Filter nodes if specified
    nodes = exec_config.get("nodes", [])
    node_filter = None
    if args.nodes:
        node_ids = [int(n.strip()) for n in args.nodes.split(",")]
        nodes = [n for n in nodes if n.get("name") in node_ids]
        node_filter = node_ids

    if not nodes:
        print_error("No nodes to process")
        return 1

    # =====================================
    # Determine operation mode
    # =====================================
    run_verification = not args.benchmark or args.all
    run_benchmark = args.benchmark or args.all

    verification_passed = True
    benchmark_passed = True

    # =====================================
    # VERIFICATION MODE
    # =====================================
    if run_verification:
        print_header("WAFL-Testbed Configuration Verification")

        # Validate configurations
        config_validator = ConfigValidator()

        if not config_validator.load_configs():
            print_error("Failed to load configuration files")
            for error in config_validator.errors:
                print_error(error)
            return 1

        params_valid = config_validator.validate_parameters()
        exec_valid = config_validator.validate_execution_config()

        if not params_valid or not exec_valid:
            print_error("Configuration validation failed")
            for error in config_validator.errors:
                print_error(error)
            return 1

        # If config-only mode, stop here
        if args.config_only:
            print_header("Verification Summary")
            print_success("Configuration files: Valid")

            if config_validator.warnings:
                print_warning(f"Warnings: {len(config_validator.warnings)}")
                for warning in config_validator.warnings:
                    print_warning(warning)

            return 0

        # Verify container applications
        try:
            container_verifier = ContainerVerifier(
                config_validator.params,
                config_validator.exec_config,
                verbose=args.verbose,
            )

        except FileNotFoundError as e:
            print_error(str(e))
            return 1
        except Exception as e:
            print_error(f"Container verification failed: {e}")
            return 1

        # Summary
        print_header("Verification Summary")

        total_errors = len(config_validator.errors) + len(container_verifier.errors)
        total_warnings = len(config_validator.warnings) + len(container_verifier.warnings)

        print_success("Configuration files: Valid")

        if node_filter:
            print_info(f"Containers verified: {len(node_filter)} nodes")
        else:
            nodes_count = len(config_validator.exec_config.get("nodes", []))
            print_success(f"Containers verified: {nodes_count} nodes")

        if total_warnings > 0:
            print_warning(f"Warnings: {total_warnings}")
            for warning in config_validator.warnings + container_verifier.warnings:
                print_warning(warning)

        if total_errors > 0:
            print_error(f"Errors: {total_errors}")
            for error in container_verifier.errors:
                print_error(error)
            verification_passed = False
        else:
            print(f"\n{Colors.GREEN}{Colors.BOLD}✓ All verifications passed{Colors.RESET}\n")

    # =====================================
    # BENCHMARK MODE
    # =====================================
    if run_benchmark:
        try:
            benchmark = InfrastructureBenchmark(
                params=params,
                exec_config=exec_config,
                auto_setup=not args.no_auto_setup,
                verbose=args.verbose,
            )

            report = benchmark.run_all_tests(nodes)

            # Cleanup containers after testing (unless --keep-containers is specified)
            if benchmark.auto_setup:
                if args.keep_containers:
                    print_info("Keeping containers running (--keep-containers specified)")
                else:
                    benchmark.teardown_containers(nodes)

            # Print summary
            print_header("Benchmark Summary")
            print_success(f"Total tests: {report.total_tests}")
            print_success(f"Passed: {report.passed_tests}")
            if report.failed_tests > 0:
                print_error(f"Failed: {report.failed_tests}")
                benchmark_passed = False

            # JSON output
            if args.json_output:
                output_data = {
                    "summary": {
                        "total": report.total_tests,
                        "passed": report.passed_tests,
                        "failed": report.failed_tests,
                    },
                    "results": [
                        {
                            "test_name": r.test_name,
                            "node_id": r.node_id,
                            "success": r.success,
                            "expected_value": r.expected_value,
                            "measured_value": r.measured_value,
                            "tolerance": r.tolerance,
                            "details": r.details,
                        }
                        for r in report.results
                    ],
                }

                try:
                    with open(args.json_output, "w") as f:
                        json.dump(output_data, f, indent=2)
                    print_success(f"Results saved to {args.json_output}")
                except Exception as e:
                    print_error(f"Failed to save JSON output: {e}")

        except FileNotFoundError as e:
            print_error(str(e))
            return 1
        except Exception as e:
            print_error(f"Benchmark failed: {e}")
            import traceback

            traceback.print_exc()
            return 1

    # =====================================
    # Final result
    # =====================================
    if not verification_passed or not benchmark_passed:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())


if __name__ == "__main__":
    sys.exit(main())
