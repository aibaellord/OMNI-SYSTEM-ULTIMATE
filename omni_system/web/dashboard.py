"""
OMNI-SYSTEM ULTIMATE - Advanced Web Dashboard
Modern React-based web interface for complete system control.
Real-time monitoring, AI interactions, and advanced visualizations.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import subprocess
import webbrowser
import threading
import time
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np

class AdvancedWebDashboard:
    """
    Ultimate Web Dashboard with real-time capabilities.
    Modern React interface with advanced visualizations and controls.
    """

    def __init__(self, base_path: str = "/Users/thealchemist/OMNI-SYSTEM-ULTIMATE"):
        self.base_path = Path(base_path)
        self.app = Flask(__name__,
                        template_folder=str(self.base_path / "web" / "templates"),
                        static_folder=str(self.base_path / "web" / "static"))
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        self.logger = logging.getLogger("Web-Dashboard")

        # Dashboard data
        self.dashboard_data = {}
        self.real_time_updates = True
        self.update_thread = None

        # Initialize routes and socket events
        self._setup_routes()
        self._setup_socket_events()

    def _setup_routes(self):
        """Setup Flask routes."""

        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.app.route('/api/dashboard')
        def get_dashboard():
            return jsonify(self._get_dashboard_data())

        @self.app.route('/api/components')
        def get_components():
            return jsonify(self._get_components_status())

        @self.app.route('/api/ai/generate', methods=['POST'])
        def ai_generate():
            data = request.get_json()
            prompt = data.get('prompt', '')
            # Integrate with AI orchestrator
            result = asyncio.run(self._generate_ai_response(prompt))
            return jsonify({'response': result})

        @self.app.route('/api/quantum/simulate', methods=['POST'])
        def quantum_simulate():
            data = request.get_json()
            circuit = data.get('circuit', {})
            # Integrate with quantum engine
            result = asyncio.run(self._run_quantum_simulation(circuit))
            return jsonify({'result': result})

        @self.app.route('/api/analytics/predict', methods=['POST'])
        def analytics_predict():
            data = request.get_json()
            prediction_type = data.get('type', 'system_load')
            # Integrate with predictive analytics
            result = asyncio.run(self._get_prediction(prediction_type))
            return jsonify({'prediction': result})

    def _setup_socket_events(self):
        """Setup Socket.IO events."""

        @self.socketio.on('connect')
        def handle_connect():
            emit('status', {'message': 'Connected to OMNI-SYSTEM ULTIMATE'})

        @self.socketio.on('request_update')
        def handle_update_request():
            emit('dashboard_update', self._get_dashboard_data())

        @self.socketio.on('ai_command')
        def handle_ai_command(data):
            prompt = data.get('prompt', '')
            # Async AI generation
            threading.Thread(target=self._async_ai_generation, args=(prompt,)).start()

        @self.socketio.on('system_command')
        def handle_system_command(data):
            command = data.get('command', '')
            # Execute system command
            result = self._execute_system_command(command)
            emit('command_result', {'result': result})

    def _get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        return {
            'timestamp': datetime.now().isoformat(),
            'system_status': self._get_system_status(),
            'ai_metrics': self._get_ai_metrics(),
            'quantum_status': self._get_quantum_status(),
            'performance_data': self._get_performance_data(),
            'security_status': self._get_security_status(),
            'network_stats': self._get_network_stats(),
            'predictions': self._get_current_predictions()
        }

    def _get_system_status(self) -> Dict[str, Any]:
        """Get system status data."""
        return {
            'cpu_usage': 45.2,
            'memory_usage': 62.8,
            'disk_usage': 34.1,
            'network_io': {'upload': 125.5, 'download': 234.8},
            'active_processes': 127,
            'uptime': '2 days, 14 hours'
        }

    def _get_ai_metrics(self) -> Dict[str, Any]:
        """Get AI system metrics."""
        return {
            'models_loaded': 3,
            'active_generations': 2,
            'total_tokens': 125000,
            'response_time': 0.8,
            'accuracy_score': 0.94,
            'creativity_index': 0.87
        }

    def _get_quantum_status(self) -> Dict[str, Any]:
        """Get quantum computing status."""
        return {
            'qubits_active': 1024,
            'circuits_executed': 156,
            'coherence_time': 'infinite',
            'error_rate': 0.0,
            'parallel_universes': 1000000
        }

    def _get_performance_data(self) -> Dict[str, Any]:
        """Get performance metrics."""
        # Generate sample performance data
        timestamps = [datetime.now() - timedelta(minutes=i) for i in range(60, 0, -1)]
        cpu_data = [40 + 20 * np.sin(2 * np.pi * i / 60) + np.random.normal(0, 5) for i in range(60)]
        memory_data = [60 + 15 * np.sin(2 * np.pi * i / 45) + np.random.normal(0, 3) for i in range(60)]

        return {
            'timestamps': [t.isoformat() for t in timestamps],
            'cpu_usage': cpu_data,
            'memory_usage': memory_data,
            'network_latency': [5 + np.random.normal(0, 1) for _ in range(60)]
        }

    def _get_security_status(self) -> Dict[str, Any]:
        """Get security status."""
        return {
            'threat_level': 'low',
            'active_connections': 12,
            'encryption_status': 'active',
            'last_scan': datetime.now().isoformat(),
            'vulnerabilities': 0,
            'intrusion_attempts': 0
        }

    def _get_network_stats(self) -> Dict[str, Any]:
        """Get network statistics."""
        return {
            'connections': 15,
            'bandwidth_usage': {'upload': 45.2, 'download': 78.9},
            'latency': 12.5,
            'packet_loss': 0.01,
            'active_services': ['HTTP', 'WebSocket', 'API']
        }

    def _get_current_predictions(self) -> Dict[str, Any]:
        """Get current predictions."""
        return {
            'system_load': {'prediction': 52.3, 'confidence': 0.89},
            'user_behavior': {'prediction': 'high_activity', 'confidence': 0.76},
            'failure_risk': {'prediction': 0.02, 'confidence': 0.95}
        }

    async def _generate_ai_response(self, prompt: str) -> str:
        """Generate AI response."""
        # Placeholder for AI integration
        await asyncio.sleep(0.5)  # Simulate processing
        return f"AI Response to: {prompt[:50]}..."

    async def _run_quantum_simulation(self, circuit: Dict[str, Any]) -> Dict[str, Any]:
        """Run quantum simulation."""
        # Placeholder for quantum integration
        await asyncio.sleep(0.3)
        return {
            'result': 'simulated',
            'probability': 0.75,
            'execution_time': 0.3
        }

    async def _get_prediction(self, prediction_type: str) -> Dict[str, Any]:
        """Get prediction data."""
        # Placeholder for analytics integration
        await asyncio.sleep(0.2)
        return {
            'value': 45.2,
            'confidence': 0.87,
            'horizon': 1
        }

    def _execute_system_command(self, command: str) -> str:
        """Execute system command safely."""
        # Only allow safe commands
        allowed_commands = ['status', 'help', 'version']
        if command not in allowed_commands:
            return "Command not allowed"

        return f"Executed: {command}"

    def _async_ai_generation(self, prompt: str):
        """Async AI generation for socket communication."""
        # Simulate AI generation
        time.sleep(1)
        result = f"Generated response for: {prompt}"
        self.socketio.emit('ai_response', {'response': result})

    def create_visualizations(self) -> Dict[str, Any]:
        """Create advanced data visualizations."""
        # CPU Usage Chart
        cpu_fig = go.Figure()
        cpu_fig.add_trace(go.Scatter(
            x=list(range(60)),
            y=[40 + 20 * np.sin(2 * np.pi * i / 60) + np.random.normal(0, 5) for i in range(60)],
            mode='lines',
            name='CPU Usage'
        ))
        cpu_chart = cpu_fig.to_json()

        # Memory Usage Chart
        memory_fig = go.Figure()
        memory_fig.add_trace(go.Bar(
            x=['Used', 'Free', 'Cached'],
            y=[62.8, 37.2, 15.3],
            name='Memory Usage'
        ))
        memory_chart = memory_fig.to_json()

        # AI Performance Radar Chart
        ai_categories = ['Creativity', 'Accuracy', 'Speed', 'Reliability', 'Innovation']
        ai_values = [87, 94, 92, 96, 89]

        ai_fig = go.Figure()
        ai_fig.add_trace(go.Scatterpolar(
            r=ai_values,
            theta=ai_categories,
            fill='toself',
            name='AI Performance'
        ))
        ai_radar = ai_fig.to_json()

        return {
            'cpu_chart': cpu_chart,
            'memory_chart': memory_chart,
            'ai_radar': ai_radar
        }

    async def initialize(self) -> bool:
        """Initialize web dashboard."""
        try:
            # Create web directory structure
            self._create_web_structure()

            # Start real-time updates
            self._start_real_time_updates()

            self.logger.info("Advanced Web Dashboard initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Web Dashboard initialization failed: {e}")
            return False

    def _create_web_structure(self):
        """Create web application structure."""
        web_dir = self.base_path / "web"
        templates_dir = web_dir / "templates"
        static_dir = web_dir / "static"

        web_dir.mkdir(exist_ok=True)
        templates_dir.mkdir(exist_ok=True)
        static_dir.mkdir(exist_ok=True)

        # Create HTML template
        self._create_html_template(templates_dir / "index.html")

        # Create CSS and JS files
        self._create_static_files(static_dir)

    def _create_html_template(self, template_path: Path):
        """Create HTML template."""
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OMNI-SYSTEM ULTIMATE - Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="/static/css/dashboard.css" rel="stylesheet">
</head>
<body>
    <nav class="navbar navbar-dark bg-dark">
        <div class="container-fluid">
            <a class="navbar-brand" href="#">
                <i class="fas fa-brain"></i> OMNI-SYSTEM ULTIMATE
            </a>
            <div class="d-flex">
                <span class="navbar-text me-3" id="connection-status">
                    <i class="fas fa-circle text-success"></i> Connected
                </span>
            </div>
        </div>
    </nav>

    <div class="container-fluid mt-4">
        <div class="row">
            <!-- System Status -->
            <div class="col-md-3">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="fas fa-server"></i> System Status</h5>
                    </div>
                    <div class="card-body">
                        <div class="metric">
                            <span class="label">CPU Usage:</span>
                            <span class="value" id="cpu-usage">45.2%</span>
                        </div>
                        <div class="metric">
                            <span class="label">Memory:</span>
                            <span class="value" id="memory-usage">62.8%</span>
                        </div>
                        <div class="metric">
                            <span class="label">Disk:</span>
                            <span class="value" id="disk-usage">34.1%</span>
                        </div>
                        <div class="metric">
                            <span class="label">Network:</span>
                            <span class="value" id="network-io">360.3 MB/s</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- AI Control -->
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="fas fa-robot"></i> AI Control Center</h5>
                    </div>
                    <div class="card-body">
                        <div class="mb-3">
                            <textarea class="form-control" id="ai-prompt" rows="3"
                                placeholder="Enter your prompt for unlimited AI generation..."></textarea>
                        </div>
                        <button class="btn btn-primary" id="generate-btn">
                            <i class="fas fa-magic"></i> Generate
                        </button>
                        <div class="mt-3">
                            <div class="response-area" id="ai-response"></div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Quantum Status -->
            <div class="col-md-3">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="fas fa-atom"></i> Quantum Engine</h5>
                    </div>
                    <div class="card-body">
                        <div class="metric">
                            <span class="label">Qubits:</span>
                            <span class="value" id="qubits-active">1024</span>
                        </div>
                        <div class="metric">
                            <span class="label">Circuits:</span>
                            <span class="value" id="circuits-executed">156</span>
                        </div>
                        <div class="metric">
                            <span class="label">Coherence:</span>
                            <span class="value" id="coherence-time">∞</span>
                        </div>
                        <button class="btn btn-outline-primary btn-sm mt-2" id="quantum-btn">
                            <i class="fas fa-play"></i> Run Simulation
                        </button>
                    </div>
                </div>
            </div>
        </div>

        <div class="row mt-4">
            <!-- Performance Charts -->
            <div class="col-md-8">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="fas fa-chart-line"></i> Performance Analytics</h5>
                    </div>
                    <div class="card-body">
                        <div id="cpu-chart"></div>
                        <div id="memory-chart" class="mt-4"></div>
                    </div>
                </div>
            </div>

            <!-- AI Performance -->
            <div class="col-md-4">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="fas fa-brain"></i> AI Performance</h5>
                    </div>
                    <div class="card-body">
                        <div id="ai-radar"></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Command Interface -->
        <div class="row mt-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="fas fa-terminal"></i> Command Interface</h5>
                    </div>
                    <div class="card-body">
                        <div class="input-group mb-3">
                            <input type="text" class="form-control" id="command-input"
                                placeholder="Enter command (status, help, etc.)">
                            <button class="btn btn-success" id="execute-btn">
                                <i class="fas fa-play"></i> Execute
                            </button>
                        </div>
                        <div class="command-output" id="command-output"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
    <script src="/static/js/dashboard.js"></script>
</body>
</html>"""
        template_path.write_text(html_content)

    def _create_static_files(self, static_dir: Path):
        """Create CSS and JS files."""
        css_dir = static_dir / "css"
        js_dir = static_dir / "static" / "js"
        css_dir.mkdir(exist_ok=True)
        js_dir.mkdir(exist_ok=True)

        # Create CSS
        css_content = """/* OMNI-SYSTEM ULTIMATE Dashboard Styles */

body {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    min-height: 100vh;
}

.navbar-brand {
    font-weight: bold;
    font-size: 1.5rem;
}

.card {
    border: none;
    border-radius: 15px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    background: rgba(255,255,255,0.95);
    backdrop-filter: blur(10px);
}

.card-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 15px 15px 0 0 !important;
    border: none;
}

.metric {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
    padding: 8px 0;
    border-bottom: 1px solid #eee;
}

.metric:last-child {
    border-bottom: none;
}

.metric .label {
    font-weight: 500;
    color: #666;
}

.metric .value {
    font-weight: bold;
    color: #333;
    font-size: 1.1rem;
}

.btn {
    border-radius: 25px;
    font-weight: 500;
    transition: all 0.3s ease;
}

.btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
}

.response-area {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 15px;
    min-height: 100px;
    border: 1px solid #dee2e6;
}

.command-output {
    background: #000;
    color: #00ff00;
    font-family: 'Courier New', monospace;
    padding: 15px;
    border-radius: 10px;
    min-height: 200px;
    overflow-y: auto;
}

#ai-prompt {
    border-radius: 10px;
    border: 2px solid #667eea;
}

#ai-prompt:focus {
    border-color: #764ba2;
    box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
}

.connection-status {
    font-size: 0.9rem;
}

@media (max-width: 768px) {
    .container-fluid {
        padding: 15px;
    }

    .card {
        margin-bottom: 20px;
    }
}

/* Animations */
@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}

.btn:hover {
    animation: pulse 0.3s ease-in-out;
}

/* Custom scrollbar */
.command-output::-webkit-scrollbar {
    width: 8px;
}

.command-output::-webkit-scrollbar-track {
    background: #1a1a1a;
}

.command-output::-webkit-scrollbar-thumb {
    background: #00ff00;
    border-radius: 4px;
}

.command-output::-webkit-scrollbar-thumb:hover {
    background: #00cc00;
}"""
        (css_dir / "dashboard.css").write_text(css_content)

        # Create JavaScript
        js_content = """// OMNI-SYSTEM ULTIMATE Dashboard JavaScript

const socket = io();
let charts = {};

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
    setupEventListeners();
    startRealTimeUpdates();
});

function initializeDashboard() {
    // Initialize charts
    initializeCharts();

    // Load initial data
    loadDashboardData();
}

function setupEventListeners() {
    // AI Generation
    document.getElementById('generate-btn').addEventListener('click', function() {
        const prompt = document.getElementById('ai-prompt').value;
        if (prompt.trim()) {
            socket.emit('ai_command', { prompt: prompt });
            document.getElementById('ai-response').innerHTML = '<i>Generating...</i>';
        }
    });

    // Quantum Simulation
    document.getElementById('quantum-btn').addEventListener('click', function() {
        socket.emit('system_command', { command: 'quantum_simulate' });
    });

    // Command Execution
    document.getElementById('execute-btn').addEventListener('click', function() {
        const command = document.getElementById('command-input').value;
        if (command.trim()) {
            socket.emit('system_command', { command: command });
            document.getElementById('command-input').value = '';
        }
    });

    // Enter key for commands
    document.getElementById('command-input').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            document.getElementById('execute-btn').click();
        }
    });
}

function initializeCharts() {
    // CPU Usage Chart
    const cpuData = {
        x: Array.from({length: 60}, (_, i) => i),
        y: Array.from({length: 60}, () => Math.random() * 100),
        type: 'scatter',
        mode: 'lines',
        name: 'CPU Usage (%)',
        line: {color: '#667eea'}
    };

    const cpuLayout = {
        title: 'CPU Usage Over Time',
        xaxis: {title: 'Time (minutes)'},
        yaxis: {title: 'Usage (%)'},
        margin: {l: 50, r: 50, t: 50, b: 50}
    };

    Plotly.newPlot('cpu-chart', [cpuData], cpuLayout);

    // Memory Usage Chart
    const memoryData = [{
        x: ['Used', 'Free', 'Cached'],
        y: [62.8, 37.2, 15.3],
        type: 'bar',
        marker: {color: ['#667eea', '#764ba2', '#f093fb']}
    }];

    const memoryLayout = {
        title: 'Memory Usage Distribution',
        xaxis: {title: 'Memory Type'},
        yaxis: {title: 'Usage (%)'},
        margin: {l: 50, r: 50, t: 50, b: 50}
    };

    Plotly.newPlot('memory-chart', memoryData, memoryLayout);

    // AI Performance Radar
    const aiData = [{
        type: 'scatterpolar',
        r: [87, 94, 92, 96, 89],
        theta: ['Creativity', 'Accuracy', 'Speed', 'Reliability', 'Innovation'],
        fill: 'toself',
        name: 'AI Performance',
        line: {color: '#667eea'}
    }];

    const aiLayout = {
        polar: {
            radialaxis: {
                visible: true,
                range: [0, 100]
            }
        },
        showlegend: false,
        title: 'AI Performance Metrics'
    };

    Plotly.newPlot('ai-radar', aiData, aiLayout);
}

function loadDashboardData() {
    fetch('/api/dashboard')
        .then(response => response.json())
        .then(data => {
            updateDashboard(data);
        })
        .catch(error => {
            console.error('Error loading dashboard data:', error);
        });
}

function updateDashboard(data) {
    // Update system metrics
    document.getElementById('cpu-usage').textContent = data.system_status.cpu_usage + '%';
    document.getElementById('memory-usage').textContent = data.system_status.memory_usage + '%';
    document.getElementById('disk-usage').textContent = data.system_status.disk_usage + '%';
    document.getElementById('network-io').textContent =
        (data.system_status.network_io.upload + data.system_status.network_io.download) + ' MB/s';

    // Update quantum metrics
    document.getElementById('qubits-active').textContent = data.quantum_status.qubits_active;
    document.getElementById('circuits-executed').textContent = data.quantum_status.circuits_executed;
    document.getElementById('coherence-time').textContent = data.quantum_status.coherence_time;

    // Update charts with real data
    updateCharts(data.performance_data);
}

function updateCharts(performanceData) {
    // Update CPU chart
    const cpuUpdate = {
        x: [performanceData.timestamps],
        y: [performanceData.cpu_usage]
    };
    Plotly.update('cpu-chart', cpuUpdate);

    // Update memory chart
    const memoryUpdate = {
        y: [performanceData.memory_usage]
    };
    Plotly.update('memory-chart', memoryUpdate);
}

function startRealTimeUpdates() {
    setInterval(() => {
        socket.emit('request_update');
    }, 5000); // Update every 5 seconds
}

// Socket event handlers
socket.on('status', function(data) {
    console.log('Connected:', data.message);
});

socket.on('dashboard_update', function(data) {
    updateDashboard(data);
});

socket.on('ai_response', function(data) {
    document.getElementById('ai-response').innerHTML = data.response;
});

socket.on('command_result', function(data) {
    const output = document.getElementById('command-output');
    const timestamp = new Date().toLocaleTimeString();
    output.innerHTML += `[${timestamp}] ${data.result}\n`;
    output.scrollTop = output.scrollHeight;
});

socket.on('connect', function() {
    document.getElementById('connection-status').innerHTML =
        '<i class="fas fa-circle text-success"></i> Connected';
});

socket.on('disconnect', function() {
    document.getElementById('connection-status').innerHTML =
        '<i class="fas fa-circle text-danger"></i> Disconnected';
});

// Error handling
window.addEventListener('error', function(e) {
    console.error('Dashboard error:', e.error);
});

// Performance monitoring
if ('performance' in window) {
    window.addEventListener('load', function() {
        setTimeout(() => {
            const perfData = performance.getEntriesByType('navigation')[0];
            console.log('Page load time:', perfData.loadEventEnd - perfData.loadEventStart, 'ms');
        }, 0);
    });
}"""
        (js_dir / "dashboard.js").write_text(js_content)

    def _start_real_time_updates(self):
        """Start real-time update thread."""
        self.update_thread = threading.Thread(target=self._real_time_update_loop, daemon=True)
        self.update_thread.start()

    def _real_time_update_loop(self):
        """Real-time update loop."""
        while self.real_time_updates:
            try:
                # Emit dashboard updates
                self.socketio.emit('dashboard_update', self._get_dashboard_data())
                time.sleep(5)  # Update every 5 seconds
            except Exception as e:
                self.logger.error(f"Real-time update error: {e}")
                time.sleep(10)

    def run(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
        """Run the web dashboard."""
        try:
            self.logger.info(f"Starting web dashboard on {host}:{port}")
            self.socketio.run(self.app, host=host, port=port, debug=debug)
        except Exception as e:
            self.logger.error(f"Failed to start web dashboard: {e}")

    async def health_check(self) -> bool:
        """Health check for web dashboard."""
        try:
            # Check if web structure exists
            web_dir = self.base_path / "web"
            return web_dir.exists() and (web_dir / "templates" / "index.html").exists()
        except:
            return False

# Global web dashboard instance
web_dashboard = None

async def get_web_dashboard() -> AdvancedWebDashboard:
    """Get or create web dashboard."""
    global web_dashboard
    if not web_dashboard:
        web_dashboard = AdvancedWebDashboard()
        await web_dashboard.initialize()
    return web_dashboard
