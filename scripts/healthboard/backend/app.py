from flask import Flask, jsonify
import psutil
import os
import subprocess
import json

from pathlib import Path

app = Flask(__name__, static_folder='../frontend/output', static_url_path='/')

# Path to docker-compose.yml
DOCKER_COMPOSE_FILE = Path(__file__).parents[3] / 'docker-compose.yml'

def run_docker_command(command: list, timeout=30) -> dict:
    """Run a docker (or docker compose) command and return structured response"""
    try:
        process = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            output, error = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            output, error = process.communicate()
            return {
                'status': 'error',
                'output': None,
                'error': f'Command timed out. output: {output} error: {error}'
            }

        if process.returncode == 0:
            return {
                'status': 'success',
                'output': output.strip(),
                'error': error.strip()
            }
        else:
            return {
                'status': 'error',
                'output': output.strip(),
                'error': error.strip()
            }
    except Exception as e:
        return {
            'status': 'error',
            'output': None,
            'error': str(e)
        }

@app.route('/')
def serve_react_app():
    """Serve the React SPA"""
    return app.send_static_file('singlepage.html')

@app.route('/api/system/ram')
def get_free_ram():
    """Get free RAM information"""
    try:
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024 ** 3)  # Convert to GB
        total_gb = memory.total / (1024 ** 3)
        
        return jsonify({
            'status': 'success',
            'data': {
                'available': round(available_gb, 2),
                'total': round(total_gb, 2),
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/system/disk')
def get_free_disk():
    """Get free disk space information"""
    try:
        # Get disk usage for the root partition (or current working directory)
        disk = psutil.disk_usage('/')
        free_gb = disk.free / (1024 ** 3)  # Convert to GB
        total_gb = disk.total / (1024 ** 3)
        
        return jsonify({
            'status': 'success',
            'data': {
                'free': round(free_gb, 2),
                'total': round(total_gb, 2),
                'percent_free': round((disk.free / disk.total) * 100, 2)
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/docker/status')
def get_docker_status():
    """Check Docker Compose status with health checks"""
    if not DOCKER_COMPOSE_FILE.exists():
        return jsonify({
            'status': 'warning',
            'message': f'Docker Compose file not found: {DOCKER_COMPOSE_FILE}',
            'running': "unknown",
            'services': []
        })

    # Check if Docker daemon is running
    docker_check = run_docker_command(['docker', 'info'])
    if docker_check['status'] != 'success':
        return jsonify({
            'status': 'error',
            'message': f'Docker daemon not accessible\n{docker_check['error']}',
            'running': "unknown",
            'services': []
        })

    # Get Docker Compose services status with health information
    result = run_docker_command([
        'docker' ,'compose', '-f', str(DOCKER_COMPOSE_FILE),
        'ps', '-a', '--format', 'json'
    ])

    if result['status'] == 'success':
        try:
            services = []

            # docker compose output is jsonl starting from 2.21.0
            for line in result['output'].splitlines():
                services.append(json.loads(line))

            # Enrich services with health status
            enriched_services = []
            for service in services:
                # Get health status for each container of the service
                health_result = run_docker_command([
                    'docker', 'compose', 'ps', '--format', '{{.Name}},{{.Health}}'
                ])

                health_status = 'unknown'
                if health_result['status'] == 'success':
                    for line in health_result['output'].strip().splitlines():
                        container_name, health = line.split(',', maxsplit=1)
                        health = health.lower()
                        if container_name != service["Name"]:
                            continue
                        if health in ['healthy', 'unhealthy', 'starting']:
                            health_status = health
                        break

                service['Health'] = health_status
                enriched_services.append(service)

            running_services = [
                service for service in enriched_services
                if service.get('State') in ['running', 'up']
            ]

            return jsonify({
                'status': 'success',
                'data': {
                    'running': "yes" if len(running_services) > 0 else "no",
                    'services': enriched_services,
                    'running_count': len(running_services),
                    'total_count': len(enriched_services)
                }
            })
        except json.JSONDecodeError as exc:
            return jsonify({
                'status': 'error',
                'message': 'Could not parse service status (check docker compose version). ' + result['output']
            })
    else:
        return jsonify({
            'status': 'error',
            'message': result['error'],
        })

@app.route('/api/docker/start', methods=['POST'])
def start_docker_compose():
    """Start Docker Compose services"""
    if not DOCKER_COMPOSE_FILE.exists():
        return jsonify({
            'status': 'error',
            'message': f'Docker Compose file not found: {DOCKER_COMPOSE_FILE}'
        }), 400

    result = run_docker_command([
        'docker', 'compose', '-f', str(DOCKER_COMPOSE_FILE), 'up', '-d'
    ])
    
    response = {
        'status': result['status'],
        'data': {
            'output': result['output'],
        },
    }

    if result['status'] != "success":
        response['message'] = result['error']
    else:
        response['data']['error'] = result['error']
    
    return jsonify(response)

@app.route('/api/docker/stop', methods=['POST'])
def stop_docker_compose():
    """Stop Docker Compose services"""
    if not DOCKER_COMPOSE_FILE.exists():
        return jsonify({
            'status': 'error',
            'message': f'Docker Compose file not found: {DOCKER_COMPOSE_FILE}'
        }), 400

    result = run_docker_command([
        'docker', 'compose', '-f', str(DOCKER_COMPOSE_FILE), 'stop'
    ])

    response = {
        'status': result['status'],
        'data': {
            'output': result['output'],
        },
    }

    if result['status'] != "success":
        response['message'] = result['error']
    else:
        response['data']['error'] = result['error']
    
    return jsonify(response)

@app.route('/api/docker/create-containers', methods=['POST'])
def create_containers():
    """Create Docker Compose containers"""
    if not DOCKER_COMPOSE_FILE.exists():
        return jsonify({
            'status': 'error',
            'message': f'Docker Compose file not found: {DOCKER_COMPOSE_FILE}'
        }), 400

    result = run_docker_command([
        'docker', 'compose', '-f', str(DOCKER_COMPOSE_FILE), 'create', '--quiet-pull'
    ], timeout=None)

    response = {
        'status': result['status'],
        'data': {
            'output': result['output'],
        },
    }

    if result['status'] != "success":
        response['message'] = result['error']
    else:
        response['data']['error'] = result['error']
    
    return jsonify(response)

@app.route('/api/docker/remove-containers', methods=['POST'])
def remove_containers():
    """Remove Docker Compose containers"""
    if not DOCKER_COMPOSE_FILE.exists():
        return jsonify({
            'status': 'error',
            'message': f'Docker Compose file not found: {DOCKER_COMPOSE_FILE}'
        }), 400

    result = run_docker_command([
        'docker', 'compose', '-f', str(DOCKER_COMPOSE_FILE), 'down', '--rmi', 'all'
    ], timeout=None)

    response = {
        'status': result['status'],
        'data': {
            'output': result['output'],
        },
    }

    if result['status'] != "success":
        response['message'] = result['error']
    else:
        response['data']['error'] = result['error']
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
