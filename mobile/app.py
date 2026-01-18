"""
OMNI-SYSTEM ULTIMATE - Mobile App Companion
Mobile-optimized web interface for remote access and control of the OMNI-SYSTEM.
Responsive design with touch controls, offline capabilities, and push notifications.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import logging
import threading
import time
import uuid
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request, session, redirect, url_for
from flask_socketio import SocketIO, emit, join_room, leave_room
import qrcode
import base64
from io import BytesIO
from cryptography.fernet import Fernet
import requests

class MobileAppCompanion:
    """
    Ultimate mobile app companion.
    Web-based mobile interface with responsive design and advanced features.
    """

    def __init__(self, base_path: str = "/Users/thealchemist/OMNI-SYSTEM-ULTIMATE"):
        self.base_path = Path(base_path)
        self.logger = logging.getLogger("Mobile-App")

        # Flask app setup
        self.app = Flask(__name__,
                        template_folder=str(self.base_path / "mobile" / "templates"),
                        static_folder=str(self.base_path / "mobile" / "static"))
        self.app.secret_key = os.urandom(24).hex()
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")

        # User sessions and authentication
        self.user_sessions = {}
        self.auth_tokens = {}

        # Device management
        self.registered_devices = {}
        self.device_tokens = {}

        # Push notifications
        self.notification_queue = asyncio.Queue()
        self.notification_thread = None

        # Offline capabilities
        self.offline_cache = {}
        self.sync_queue = asyncio.Queue()

        # Real-time updates
        self.real_time_data = {}
        self.update_subscriptions = {}

        # Security
        self.encryption_key = Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)

        # Setup routes and socket events
        self._setup_routes()
        self._setup_socket_events()

    def _setup_routes(self):
        """Setup Flask routes."""

        @self.app.route('/')
        def index():
            if 'user_id' not in session:
                return redirect(url_for('login'))
            return render_template('mobile_app.html')

        @self.app.route('/login', methods=['GET', 'POST'])
        def login():
            if request.method == 'POST':
                username = request.form.get('username')
                password = request.form.get('password')

                # Simple authentication (in production, use proper auth)
                if username == 'admin' and password == 'admin':
                    user_id = str(uuid.uuid4())
                    session['user_id'] = user_id
                    session['username'] = username
                    self.user_sessions[user_id] = {
                        'username': username,
                        'login_time': datetime.now().isoformat(),
                        'device_info': request.headers.get('User-Agent', '')
                    }
                    return redirect(url_for('index'))

                return render_template('login.html', error='Invalid credentials')

            return render_template('login.html')

        @self.app.route('/logout')
        def logout():
            user_id = session.get('user_id')
            if user_id and user_id in self.user_sessions:
                del self.user_sessions[user_id]
            session.clear()
            return redirect(url_for('login'))

        @self.app.route('/register-device', methods=['POST'])
        def register_device():
            if 'user_id' not in session:
                return jsonify({'error': 'Not authenticated'}), 401

            device_info = request.get_json()
            device_id = str(uuid.uuid4())
            device_token = os.urandom(32).hex()

            self.registered_devices[device_id] = {
                'user_id': session['user_id'],
                'device_info': device_info,
                'token': device_token,
                'registered_at': datetime.now().isoformat(),
                'status': 'active'
            }

            self.device_tokens[device_token] = device_id

            # Generate QR code for device pairing
            qr_data = f"omni://{device_token}"
            qr_code = self._generate_qr_code(qr_data)

            return jsonify({
                'device_id': device_id,
                'token': device_token,
                'qr_code': qr_code,
                'status': 'registered'
            })

        @self.app.route('/api/dashboard')
        def get_dashboard():
            if 'user_id' not in session:
                return jsonify({'error': 'Not authenticated'}), 401

            return jsonify(self._get_mobile_dashboard_data())

        @self.app.route('/api/system/control', methods=['POST'])
        def system_control():
            if 'user_id' not in session:
                return jsonify({'error': 'Not authenticated'}), 401

            data = request.get_json()
            command = data.get('command')
            parameters = data.get('parameters', {})

            # Execute system command (placeholder)
            result = self._execute_mobile_command(command, parameters)

            return jsonify({'result': result})

        @self.app.route('/api/notifications')
        def get_notifications():
            if 'user_id' not in session:
                return jsonify({'error': 'Not authenticated'}), 401

            user_notifications = [
                n for n in self._get_user_notifications(session['user_id'])
                if not n.get('read', False)
            ]
            return jsonify({'notifications': user_notifications})

        @self.app.route('/api/offline/sync', methods=['POST'])
        def offline_sync():
            if 'user_id' not in session:
                return jsonify({'error': 'Not authenticated'}), 401

            sync_data = request.get_json()
            # Process offline sync data (placeholder)
            result = self._process_offline_sync(session['user_id'], sync_data)

            return jsonify({'sync_result': result})

        @self.app.route('/api/qr-pairing/<token>')
        def qr_pairing(token):
            if token in self.device_tokens:
                device_id = self.device_tokens[token]
                device = self.registered_devices[device_id]
                return jsonify({
                    'paired': True,
                    'device_id': device_id,
                    'device_info': device['device_info']
                })
            return jsonify({'paired': False, 'error': 'Invalid token'}), 404

    def _setup_socket_events(self):
        """Setup Socket.IO events."""

        @self.socketio.on('connect')
        def handle_connect():
            user_id = session.get('user_id')
            if user_id:
                join_room(user_id)
                emit('status', {'message': 'Connected to OMNI-SYSTEM Mobile'})

        @self.socketio.on('disconnect')
        def handle_disconnect():
            user_id = session.get('user_id')
            if user_id:
                leave_room(user_id)

        @self.socketio.on('subscribe_updates')
        def handle_subscribe_updates(data):
            user_id = session.get('user_id')
            if user_id:
                subscription_type = data.get('type', 'dashboard')
                if user_id not in self.update_subscriptions:
                    self.update_subscriptions[user_id] = []
                if subscription_type not in self.update_subscriptions[user_id]:
                    self.update_subscriptions[user_id].append(subscription_type)

        @self.socketio.on('mobile_command')
        def handle_mobile_command(data):
            user_id = session.get('user_id')
            if user_id:
                command = data.get('command')
                result = self._execute_mobile_command(command, data.get('parameters', {}))
                emit('command_result', {'result': result})

        @self.socketio.on('mark_notification_read')
        def handle_mark_read(data):
            user_id = session.get('user_id')
            if user_id:
                notification_id = data.get('notification_id')
                self._mark_notification_read(user_id, notification_id)

    def _generate_qr_code(self, data: str) -> str:
        """Generate QR code for device pairing."""
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(data)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return f"data:image/png;base64,{img_str}"

    def _get_mobile_dashboard_data(self) -> Dict[str, Any]:
        """Get mobile-optimized dashboard data."""
        return {
            'timestamp': datetime.now().isoformat(),
            'system_status': {
                'cpu': 45.2,
                'memory': 62.8,
                'battery': 78,
                'network': 'WiFi',
                'location': 'Home'
            },
            'quick_actions': [
                {'id': 'lights', 'name': 'Lights', 'icon': '💡', 'status': 'on'},
                {'id': 'thermostat', 'name': 'Climate', 'icon': '🌡️', 'status': '22°C'},
                {'id': 'security', 'name': 'Security', 'icon': '🔒', 'status': 'armed'},
                {'id': 'music', 'name': 'Music', 'icon': '🎵', 'status': 'playing'}
            ],
            'recent_activity': [
                {'time': '2 min ago', 'action': 'Lights turned on', 'icon': '💡'},
                {'time': '5 min ago', 'action': 'Temperature adjusted', 'icon': '🌡️'},
                {'time': '10 min ago', 'action': 'Security armed', 'icon': '🔒'}
            ],
            'notifications': self._get_user_notifications(session.get('user_id', ''))[:5]
        }

    def _execute_mobile_command(self, command: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute mobile command."""
        command_handlers = {
            'toggle_light': self._toggle_light,
            'set_temperature': self._set_temperature,
            'arm_security': self._arm_security,
            'play_music': self._play_music,
            'get_weather': self._get_weather,
            'send_notification': self._send_mobile_notification
        }

        if command in command_handlers:
            return command_handlers[command](parameters)
        else:
            return {'error': f'Unknown command: {command}'}

    def _toggle_light(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Toggle light state."""
        light_id = params.get('light_id', 'main')
        new_state = params.get('state', 'toggle')
        return {
            'command': 'toggle_light',
            'light_id': light_id,
            'new_state': 'on' if new_state == 'on' else 'off',
            'success': True
        }

    def _set_temperature(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Set temperature."""
        temperature = params.get('temperature', 22)
        return {
            'command': 'set_temperature',
            'temperature': temperature,
            'unit': 'celsius',
            'success': True
        }

    def _arm_security(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Arm/disarm security system."""
        action = params.get('action', 'arm')
        return {
            'command': 'arm_security',
            'action': action,
            'success': True
        }

    def _play_music(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Control music playback."""
        action = params.get('action', 'play')
        track = params.get('track', 'current')
        return {
            'command': 'play_music',
            'action': action,
            'track': track,
            'success': True
        }

    def _get_weather(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get weather information."""
        return {
            'command': 'get_weather',
            'location': 'Current Location',
            'temperature': 22,
            'condition': 'Sunny',
            'humidity': 65,
            'wind_speed': 5
        }

    def _send_mobile_notification(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send mobile notification."""
        title = params.get('title', 'OMNI-SYSTEM')
        message = params.get('message', 'Notification from OMNI-SYSTEM')
        user_id = session.get('user_id')

        if user_id:
            self._add_notification(user_id, title, message, 'info')

        return {
            'command': 'send_notification',
            'title': title,
            'message': message,
            'success': True
        }

    def _get_user_notifications(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user notifications."""
        # Mock notifications
        return [
            {
                'id': '1',
                'title': 'System Update',
                'message': 'OMNI-SYSTEM has been updated to version 1.1.0',
                'type': 'info',
                'timestamp': datetime.now().isoformat(),
                'read': False
            },
            {
                'id': '2',
                'title': 'Security Alert',
                'message': 'Motion detected in living room',
                'type': 'warning',
                'timestamp': (datetime.now() - timedelta(minutes=5)).isoformat(),
                'read': False
            }
        ]

    def _add_notification(self, user_id: str, title: str, message: str, notification_type: str = 'info'):
        """Add notification for user."""
        notification = {
            'id': str(uuid.uuid4()),
            'title': title,
            'message': message,
            'type': notification_type,
            'timestamp': datetime.now().isoformat(),
            'read': False
        }

        # In a real implementation, store in database
        # For now, emit via socket
        self.socketio.emit('notification', notification, room=user_id)

    def _mark_notification_read(self, user_id: str, notification_id: str):
        """Mark notification as read."""
        # In a real implementation, update database
        pass

    def _process_offline_sync(self, user_id: str, sync_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process offline sync data."""
        # Process sync data (placeholder)
        return {
            'synced_items': len(sync_data.get('items', [])),
            'conflicts_resolved': 0,
            'new_data_pulled': 5,
            'success': True
        }

    def create_mobile_templates(self):
        """Create mobile app templates."""
        template_dir = self.base_path / "mobile" / "templates"
        static_dir = self.base_path / "mobile" / "static"
        css_dir = static_dir / "css"
        js_dir = static_dir / "js"

        template_dir.mkdir(parents=True, exist_ok=True)
        css_dir.mkdir(parents=True, exist_ok=True)
        js_dir.mkdir(parents=True, exist_ok=True)

        # Create login template
        login_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OMNI-SYSTEM Mobile - Login</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="/static/css/mobile.css" rel="stylesheet">
</head>
<body class="mobile-login">
    <div class="container">
        <div class="row justify-content-center">
            <div class="col-md-6 col-lg-4">
                <div class="login-card">
                    <div class="text-center mb-4">
                        <h2 class="text-primary">🔮</h2>
                        <h3>OMNI-SYSTEM</h3>
                        <p class="text-muted">Ultimate Mobile Companion</p>
                    </div>
                    <form method="POST">
                        <div class="mb-3">
                            <input type="text" class="form-control" name="username" placeholder="Username" required>
                        </div>
                        <div class="mb-3">
                            <input type="password" class="form-control" name="password" placeholder="Password" required>
                        </div>
                        {% if error %}
                        <div class="alert alert-danger">{{ error }}</div>
                        {% endif %}
                        <button type="submit" class="btn btn-primary w-100">Login</button>
                    </form>
                </div>
            </div>
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>"""
        (template_dir / "login.html").write_text(login_html)

        # Create main mobile app template
        mobile_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OMNI-SYSTEM Mobile</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <link href="/static/css/mobile.css" rel="stylesheet">
</head>
<body class="mobile-app">
    <!-- Header -->
    <nav class="navbar navbar-dark bg-primary fixed-top">
        <div class="container-fluid">
            <a class="navbar-brand" href="#">
                <i class="fas fa-brain"></i> OMNI
            </a>
            <div class="d-flex">
                <button class="btn btn-link text-white" id="notification-btn">
                    <i class="fas fa-bell"></i>
                    <span class="badge bg-danger" id="notification-badge" style="display: none;">0</span>
                </button>
                <button class="btn btn-link text-white" onclick="logout()">
                    <i class="fas fa-sign-out-alt"></i>
                </button>
            </div>
        </div>
    </nav>

    <!-- Main Content -->
    <div class="container-fluid mt-5 pt-3">
        <!-- System Status -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card status-card">
                    <div class="card-body">
                        <div class="row text-center">
                            <div class="col-3">
                                <div class="status-item">
                                    <i class="fas fa-microchip text-primary"></i>
                                    <div class="value" id="cpu-status">45%</div>
                                    <div class="label">CPU</div>
                                </div>
                            </div>
                            <div class="col-3">
                                <div class="status-item">
                                    <i class="fas fa-memory text-success"></i>
                                    <div class="value" id="memory-status">63%</div>
                                    <div class="label">RAM</div>
                                </div>
                            </div>
                            <div class="col-3">
                                <div class="status-item">
                                    <i class="fas fa-battery-half text-warning"></i>
                                    <div class="value" id="battery-status">78%</div>
                                    <div class="label">Battery</div>
                                </div>
                            </div>
                            <div class="col-3">
                                <div class="status-item">
                                    <i class="fas fa-wifi text-info"></i>
                                    <div class="value" id="network-status">WiFi</div>
                                    <div class="label">Network</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Quick Actions -->
        <div class="row mb-4">
            <div class="col-12">
                <h5 class="mb-3">Quick Actions</h5>
                <div class="row" id="quick-actions">
                    <!-- Actions loaded dynamically -->
                </div>
            </div>
        </div>

        <!-- Recent Activity -->
        <div class="row mb-4">
            <div class="col-12">
                <h5 class="mb-3">Recent Activity</h5>
                <div class="card">
                    <div class="list-group list-group-flush" id="recent-activity">
                        <!-- Activity loaded dynamically -->
                    </div>
                </div>
            </div>
        </div>

        <!-- Control Panel -->
        <div class="row">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">Control Panel</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-6 mb-3">
                                <button class="btn btn-outline-primary w-100" onclick="sendCommand('toggle_light')">
                                    <i class="fas fa-lightbulb"></i><br>Toggle Lights
                                </button>
                            </div>
                            <div class="col-6 mb-3">
                                <button class="btn btn-outline-success w-100" onclick="sendCommand('arm_security')">
                                    <i class="fas fa-shield-alt"></i><br>Security
                                </button>
                            </div>
                            <div class="col-6 mb-3">
                                <button class="btn btn-outline-info w-100" onclick="sendCommand('play_music')">
                                    <i class="fas fa-music"></i><br>Music
                                </button>
                            </div>
                            <div class="col-6 mb-3">
                                <button class="btn btn-outline-warning w-100" onclick="sendCommand('get_weather')">
                                    <i class="fas fa-cloud-sun"></i><br>Weather
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Notifications Modal -->
    <div class="modal fade" id="notificationsModal" tabindex="-1">
        <div class="modal-dialog modal-dialog-centered">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">Notifications</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body" id="notifications-list">
                    <!-- Notifications loaded here -->
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
    <script src="/static/js/mobile.js"></script>
</body>
</html>"""
        (template_dir / "mobile_app.html").write_text(mobile_html)

        # Create CSS
        mobile_css = """/* OMNI-SYSTEM Mobile App Styles */

body.mobile-login {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    display: flex;
    align-items: center;
}

body.mobile-app {
    background-color: #f8f9fa;
    padding-bottom: 80px; /* Space for bottom nav */
}

.login-card {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 15px;
    padding: 2rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    backdrop-filter: blur(10px);
}

.status-card {
    border: none;
    border-radius: 15px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
}

.status-item {
    padding: 1rem 0;
}

.status-item i {
    font-size: 1.5rem;
    margin-bottom: 0.5rem;
    display: block;
}

.status-item .value {
    font-size: 1.2rem;
    font-weight: bold;
    color: #333;
}

.status-item .label {
    font-size: 0.8rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.quick-action-btn {
    background: white;
    border: none;
    border-radius: 15px;
    padding: 1.5rem 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    text-align: center;
    transition: transform 0.2s;
}

.quick-action-btn:hover {
    transform: translateY(-2px);
}

.quick-action-btn .icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
    display: block;
}

.quick-action-btn .name {
    font-size: 0.9rem;
    font-weight: 500;
    color: #333;
}

.quick-action-btn .status {
    font-size: 0.8rem;
    color: #666;
}

.activity-item {
    padding: 1rem;
    border-bottom: 1px solid #eee;
}

.activity-item:last-child {
    border-bottom: none;
}

.activity-item .icon {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background: #f8f9fa;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-right: 1rem;
}

.activity-item .content {
    flex: 1;
}

.activity-item .action {
    font-weight: 500;
    color: #333;
    margin-bottom: 0.25rem;
}

.activity-item .time {
    font-size: 0.8rem;
    color: #666;
}

.btn {
    border-radius: 10px;
    font-weight: 500;
}

.card {
    border: none;
    border-radius: 15px;
    box-shadow: 0 3px 10px rgba(0,0,0,0.1);
}

.navbar {
    border-radius: 0 0 15px 15px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

/* Notification badge */
#notification-badge {
    position: absolute;
    top: 5px;
    right: 5px;
    font-size: 0.7rem;
    padding: 0.2rem 0.4rem;
}

/* Responsive adjustments */
@media (max-width: 576px) {
    .container-fluid {
        padding: 0 15px;
    }

    .status-item {
        padding: 0.5rem 0;
    }

    .quick-action-btn {
        padding: 1rem 0.5rem;
    }
}

/* Dark mode support */
@media (prefers-color-scheme: dark) {
    body.mobile-app {
        background-color: #1a1a1a;
        color: #ffffff;
    }

    .card {
        background-color: #2d2d2d;
        color: #ffffff;
    }

    .status-item .label {
        color: #cccccc;
    }

    .quick-action-btn {
        background: #2d2d2d;
        color: #ffffff;
    }

    .activity-item {
        border-bottom-color: #404040;
    }
}

/* Touch-friendly interactions */
.btn, .quick-action-btn {
    min-height: 44px; /* iOS touch target minimum */
}

/* Loading animation */
.loading {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid rgba(255,255,255,.3);
    border-radius: 50%;
    border-top-color: #fff;
    animation: spin 1s ease-in-out infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}"""
        (css_dir / "mobile.css").write_text(mobile_css)

        # Create JavaScript
        mobile_js = """// OMNI-SYSTEM Mobile App JavaScript

const socket = io();
let dashboardData = {};
let notifications = [];

// Initialize app
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
    startRealTimeUpdates();
});

function initializeApp() {
    loadDashboardData();
    loadNotifications();
}

function setupEventListeners() {
    // Notification button
    document.getElementById('notification-btn').addEventListener('click', function() {
        showNotificationsModal();
    });

    // Service worker registration for PWA
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('/static/js/sw.js')
            .then(registration => {
                console.log('Service Worker registered');
            })
            .catch(error => {
                console.log('Service Worker registration failed');
            });
    }
}

function loadDashboardData() {
    fetch('/api/dashboard')
        .then(response => response.json())
        .then(data => {
            dashboardData = data;
            updateDashboardUI(data);
        })
        .catch(error => {
            console.error('Error loading dashboard data:', error);
        });
}

function updateDashboardUI(data) {
    // Update system status
    document.getElementById('cpu-status').textContent = data.system_status.cpu + '%';
    document.getElementById('memory-status').textContent = data.system_status.memory + '%';
    document.getElementById('battery-status').textContent = data.system_status.battery + '%';
    document.getElementById('network-status').textContent = data.system_status.network;

    // Update quick actions
    updateQuickActions(data.quick_actions);

    // Update recent activity
    updateRecentActivity(data.recent_activity);
}

function updateQuickActions(actions) {
    const container = document.getElementById('quick-actions');
    container.innerHTML = '';

    actions.forEach(action => {
        const col = document.createElement('div');
        col.className = 'col-6';

        const btn = document.createElement('button');
        btn.className = 'quick-action-btn w-100';
        btn.onclick = () => executeQuickAction(action.id);

        btn.innerHTML = `
            <div class="icon">${action.icon}</div>
            <div class="name">${action.name}</div>
            <div class="status">${action.status}</div>
        `;

        col.appendChild(btn);
        container.appendChild(col);
    });
}

function updateRecentActivity(activities) {
    const container = document.getElementById('recent-activity');
    container.innerHTML = '';

    activities.forEach(activity => {
        const item = document.createElement('div');
        item.className = 'activity-item d-flex align-items-center';

        item.innerHTML = `
            <div class="icon">
                ${activity.icon}
            </div>
            <div class="content">
                <div class="action">${activity.action}</div>
                <div class="time">${activity.time}</div>
            </div>
        `;

        container.appendChild(item);
    });
}

function executeQuickAction(actionId) {
    // Add loading state
    const btn = event.target.closest('.quick-action-btn');
    const originalContent = btn.innerHTML;
    btn.innerHTML = '<div class="loading"></div>';
    btn.disabled = true;

    // Execute action
    sendCommand(actionId)
        .finally(() => {
            // Restore original content
            btn.innerHTML = originalContent;
            btn.disabled = false;
        });
}

function sendCommand(command, parameters = {}) {
    return fetch('/api/system/control', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            command: command,
            parameters: parameters
        })
    })
    .then(response => response.json())
    .then(data => {
        showToast('Command executed successfully', 'success');
        return data;
    })
    .catch(error => {
        showToast('Command failed', 'error');
        console.error('Command error:', error);
        throw error;
    });
}

function loadNotifications() {
    fetch('/api/notifications')
        .then(response => response.json())
        .then(data => {
            notifications = data.notifications;
            updateNotificationBadge();
        })
        .catch(error => {
            console.error('Error loading notifications:', error);
        });
}

function updateNotificationBadge() {
    const badge = document.getElementById('notification-badge');
    const unreadCount = notifications.filter(n => !n.read).length;

    if (unreadCount > 0) {
        badge.textContent = unreadCount;
        badge.style.display = 'inline';
    } else {
        badge.style.display = 'none';
    }
}

function showNotificationsModal() {
    const modal = new bootstrap.Modal(document.getElementById('notificationsModal'));
    const list = document.getElementById('notifications-list');

    list.innerHTML = notifications.length > 0 ?
        notifications.map(n => `
            <div class="notification-item p-3 border-bottom" data-id="${n.id}">
                <div class="d-flex justify-content-between align-items-start">
                    <div class="flex-grow-1">
                        <h6 class="mb-1">${n.title}</h6>
                        <p class="mb-1 text-muted">${n.message}</p>
                        <small class="text-muted">${formatTime(n.timestamp)}</small>
                    </div>
                    ${!n.read ? '<span class="badge bg-primary">New</span>' : ''}
                </div>
            </div>
        `).join('') :
        '<p class="text-center text-muted p-3">No notifications</p>';

    modal.show();
}

function formatTime(timestamp) {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now - date;
    const minutes = Math.floor(diff / 60000);

    if (minutes < 1) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;
    if (minutes < 1440) return `${Math.floor(minutes / 60)}h ago`;
    return `${Math.floor(minutes / 1440)}d ago`;
}

function showToast(message, type = 'info') {
    // Simple toast implementation
    const toast = document.createElement('div');
    toast.className = `alert alert-${type} position-fixed`;
    toast.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    toast.innerHTML = `
        ${message}
        <button type="button" class="btn-close" onclick="this.parentElement.remove()"></button>
    `;

    document.body.appendChild(toast);

    setTimeout(() => {
        if (toast.parentElement) {
            toast.remove();
        }
    }, 5000);
}

function logout() {
    fetch('/logout')
        .then(() => {
            window.location.href = '/login';
        });
}

function startRealTimeUpdates() {
    socket.emit('subscribe_updates', { type: 'dashboard' });

    // Refresh data every 30 seconds
    setInterval(() => {
        loadDashboardData();
        loadNotifications();
    }, 30000);
}

// Socket event handlers
socket.on('connect', function() {
    console.log('Connected to OMNI-SYSTEM');
});

socket.on('disconnect', function() {
    console.log('Disconnected from OMNI-SYSTEM');
});

socket.on('dashboard_update', function(data) {
    updateDashboardUI(data);
});

socket.on('notification', function(notification) {
    notifications.unshift(notification);
    updateNotificationBadge();
    showToast(`${notification.title}: ${notification.message}`, 'info');
});

socket.on('command_result', function(data) {
    console.log('Command result:', data);
});

// PWA Install Prompt
let deferredPrompt;

window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    deferredPrompt = e;

    // Show install button
    const installBtn = document.createElement('button');
    installBtn.className = 'btn btn-primary position-fixed';
    installBtn.style.cssText = 'bottom: 20px; left: 20px; z-index: 9999;';
    installBtn.innerHTML = 'Install App';
    installBtn.onclick = () => {
        deferredPrompt.prompt();
        deferredPrompt.userChoice.then((choiceResult) => {
            if (choiceResult.outcome === 'accepted') {
                console.log('User accepted the install prompt');
            }
            deferredPrompt = null;
            installBtn.remove();
        });
    };

    document.body.appendChild(installBtn);
});

// Offline detection
window.addEventListener('online', function() {
    showToast('Back online', 'success');
    // Sync offline data
    syncOfflineData();
});

window.addEventListener('offline', function() {
    showToast('You are offline', 'warning');
});

function syncOfflineData() {
    // Get offline data from localStorage
    const offlineData = JSON.parse(localStorage.getItem('omni_offline_data') || '[]');

    if (offlineData.length > 0) {
        fetch('/api/offline/sync', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ items: offlineData })
        })
        .then(response => response.json())
        .then(data => {
            if (data.sync_result.success) {
                localStorage.removeItem('omni_offline_data');
                showToast('Offline data synced', 'success');
            }
        })
        .catch(error => {
            console.error('Sync error:', error);
        });
    }
}

// Error handling
window.addEventListener('error', function(e) {
    console.error('Mobile app error:', e.error);
});

window.addEventListener('unhandledrejection', function(e) {
    console.error('Unhandled promise rejection:', e.reason);
});"""
        (js_dir / "mobile.js").write_text(mobile_js)

        # Create service worker for PWA
        sw_js = """// OMNI-SYSTEM Service Worker for PWA

const CACHE_NAME = 'omni-system-mobile-v1';
const urlsToCache = [
    '/',
    '/static/css/mobile.css',
    '/static/js/mobile.js',
    'https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css',
    'https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
];

// Install event
self.addEventListener('install', function(event) {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(function(cache) {
                return cache.addAll(urlsToCache);
            })
    );
});

// Fetch event
self.addEventListener('fetch', function(event) {
    event.respondWith(
        caches.match(event.request)
            .then(function(response) {
                // Return cached version or fetch from network
                return response || fetch(event.request);
            }
        )
    );
});

// Activate event
self.addEventListener('activate', function(event) {
    event.waitUntil(
        caches.keys().then(function(cacheNames) {
            return Promise.all(
                cacheNames.map(function(cacheName) {
                    if (cacheName !== CACHE_NAME) {
                        return caches.delete(cacheName);
                    }
                })
            );
        })
    );
});

// Background sync for offline actions
self.addEventListener('sync', function(event) {
    if (event.tag === 'background-sync') {
        event.waitUntil(doBackgroundSync());
    }
});

function doBackgroundSync() {
    // Perform background sync operations
    return fetch('/api/offline/sync', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            items: JSON.parse(localStorage.getItem('omni_offline_data') || '[]')
        })
    });
}

// Push notifications
self.addEventListener('push', function(event) {
    const options = {
        body: event.data ? event.data.text() : 'New notification from OMNI-SYSTEM',
        icon: '/static/icons/icon-192x192.png',
        badge: '/static/icons/badge-72x72.png',
        vibrate: [100, 50, 100],
        data: {
            dateOfArrival: Date.now(),
            primaryKey: 1
        },
        actions: [
            {
                action: 'explore',
                title: 'View Details',
                icon: '/static/icons/action-view-128x128.png'
            },
            {
                action: 'close',
                title: 'Close',
                icon: '/static/icons/action-close-128x128.png'
            }
        ]
    };

    event.waitUntil(
        self.registration.showNotification('OMNI-SYSTEM', options)
    );
});

// Notification click event
self.addEventListener('notificationclick', function(event) {
    event.notification.close();

    if (event.action === 'explore') {
        event.waitUntil(
            clients.openWindow('/')
        );
    }
});"""
        (js_dir / "sw.js").write_text(sw_js)

    async def initialize(self) -> bool:
        """Initialize mobile app companion."""
        try:
            # Create mobile app structure
            self.create_mobile_templates()

            # Start notification processor
            self._start_notification_processor()

            self.logger.info("Advanced Mobile App Companion initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Mobile app companion initialization failed: {e}")
            return False

    def _start_notification_processor(self):
        """Start notification processing thread."""
        self.notification_thread = threading.Thread(target=self._notification_processor_loop, daemon=True)
        self.notification_thread.start()

    def _notification_processor_loop(self):
        """Notification processing loop."""
        while True:
            try:
                # Process notification queue
                while not self.notification_queue.empty():
                    notification = self.notification_queue.get_nowait()
                    self._send_push_notification(notification)
                    self.notification_queue.task_done()

                time.sleep(1)
            except Exception as e:
                self.logger.error(f"Notification processing error: {e}")
                time.sleep(5)

    def _send_push_notification(self, notification: Dict[str, Any]):
        """Send push notification."""
        # Push notification implementation (placeholder)
        print(f"PUSH NOTIFICATION: {notification}")

    def run(self, host: str = '0.0.0.0', port: int = 8082, debug: bool = False):
        """Run the mobile app companion."""
        try:
            self.logger.info(f"Starting Mobile App Companion on {host}:{port}")
            self.socketio.run(self.app, host=host, port=port, debug=debug)
        except Exception as e:
            self.logger.error(f"Failed to start mobile app companion: {e}")

    async def health_check(self) -> bool:
        """Health check for mobile app companion."""
        try:
            # Check if templates exist
            template_dir = self.base_path / "mobile" / "templates"
            return template_dir.exists() and (template_dir / "mobile_app.html").exists()
        except:
            return False

# Global mobile app companion instance
mobile_app = None

async def get_mobile_app() -> AdvancedMobileAppCompanion:
    """Get or create mobile app companion."""
    global mobile_app
    if not mobile_app:
        mobile_app = AdvancedMobileAppCompanion()
        await mobile_app.initialize()
    return mobile_app
