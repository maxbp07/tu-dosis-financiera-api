#!/usr/bin/env python3
"""
Servidor Flask para generación de videos via API.
Este servidor expone el script newsletter_video_gen.py como una API REST.
"""

import os
import sys
import subprocess
import uuid
import logging
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, send_file, send_from_directory

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuración
PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "videos"

# Asegurar que existen los directorios
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Variable para almacenar el proceso actual en ejecución
current_process = None


@app.route('/health', methods=['GET'])
def health():
    """Endpoint de health check."""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "project_root": str(PROJECT_ROOT)
    })


@app.route('/generate-video', methods=['POST'])
def generate_video():
    """
    Endpoint principal para generar videos.
    Recibe JSON con parámetros y ejecuta newsletter_video_gen.py
    """
    global current_process

    try:
        # Parsear el JSON de entrada
        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "error": "No se recibieron datos JSON"
            }), 400

        # Extraer parámetros
        guion = data.get('guion', '')
        titulo = data.get('titulo', 'Video sin título')
        hook = data.get('hook', '')
        categoria = data.get('categoria', '')
        cta_midroll = data.get('cta_midroll', '')
        parte = data.get('parte', '')

        # Validar parámetros obligatorios
        if not guion:
            return jsonify({
                "success": False,
                "error": "El parámetro 'guion' es obligatorio"
            }), 400

        if not titulo:
            return jsonify({
                "success": False,
                "error": "El parámetro 'titulo' es obligatorio"
            }), 400

        logger.info(f"Solicitud recibida: título='{titulo}'")

        # Verificar si ya hay un proceso en ejecución
        if current_process and current_process.poll() is None:
            return jsonify({
                "success": False,
                "error": "Ya hay un video generándose. Espere a que termine."
            }), 409

        # Generar nombre único para el archivo de salida
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"{timestamp}_video.mp4"
        output_path = OUTPUTS_DIR / output_filename

        # Construir el comando para ejecutar el script
        script_path = SCRIPTS_DIR / "newsletter_video_gen.py"

        cmd = [
            sys.executable,
            str(script_path),
            "--guion", guion,
            "--titulo", titulo,
            "--output", str(output_path)
        ]

        # Añadir parámetros opcionales
        if hook:
            cmd.extend(["--hook", hook])
        if categoria:
            cmd.extend(["--categoria", categoria])
        if cta_midroll:
            cmd.extend(["--cta_midroll", cta_midroll])
        if parte:
            cmd.extend(["--parte", parte])

        logger.info(f"Ejecutando comando: {' '.join(cmd)}")

        # Ejecutar el script en modo asíncrono
        current_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(PROJECT_ROOT)
        )

        return jsonify({
            "success": True,
            "message": "Video generation started",
            "output_filename": output_filename,
            "output_path": str(output_path),
            "timestamp": datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error en /generate-video: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"Error interno del servidor: {str(e)}"
        }), 500


@app.route('/check-status', methods=['GET'])
def check_status():
    """
    Verifica el estado de la generación de video actual.
    """
    global current_process

    if not current_process:
        return jsonify({
            "status": "idle",
            "message": "No hay generación en curso"
        }), 200

    if current_process.poll() is None:
        return jsonify({
            "status": "processing",
            "message": "Generando video..."
        }), 200

    # El proceso terminó
    returncode = current_process.returncode
    stdout, stderr = current_process.communicate()

    if returncode == 0:
        return jsonify({
            "status": "completed",
            "message": "Video generado exitosamente",
            "returncode": returncode
        }), 200
    else:
        return jsonify({
            "status": "error",
            "message": f"Error en la generación: {returncode}",
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr
        }), 500


@app.route('/list-videos', methods=['GET'])
def list_videos():
    """
    Lista todos los videos generados.
    """
    try:
        videos = []
        for video_file in OUTPUTS_DIR.glob("*.mp4"):
            stat = video_file.stat()
            videos.append({
                "filename": video_file.name,
                "path": str(video_file),
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })

        # Ordenar por fecha de modificación (más reciente primero)
        videos.sort(key=lambda x: x['modified'], reverse=True)

        return jsonify({
            "success": True,
            "videos": videos,
            "count": len(videos)
        }), 200

    except Exception as e:
        logger.error(f"Error en /list-videos: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"Error listando videos: {str(e)}"
        }), 500


@app.route('/download/<filename>', methods=['GET'])
def download_video(filename):
    """
    Descarga un video específico.
    """
    try:
        video_path = OUTPUTS_DIR / filename

        if not video_path.exists():
            return jsonify({
                "success": False,
                "error": f"Video no encontrado: {filename}"
            }), 404

        return send_from_directory(str(OUTPUTS_DIR), filename)

    except Exception as e:
        logger.error(f"Error en /download: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"Error descargando video: {str(e)}"
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Manejador de errores 404."""
    return jsonify({
        "success": False,
        "error": "Recurso no encontrado"
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Manejador de errores 500."""
    logger.error(f"Error interno: {str(error)}", exc_info=True)
    return jsonify({
        "success": False,
        "error": "Error interno del servidor"
    }), 500


if __name__ == '__main__':
    print("=" * 60)
    print("Servidor de Generación de Videos - API REST")
    print("=" * 60)
    print(f"Directorio del proyecto: {PROJECT_ROOT}")
    print(f"Directorio de scripts: {SCRIPTS_DIR}")
    print(f"Directorio de outputs: {OUTPUTS_DIR}")
    print(f"Servidor iniciado en: http://0.0.0.0:8000")
    print("=" * 60)

    # Ejecutar servidor
    app.run(
        host='0.0.0.0',
        port=8000,
        debug=False,
        threaded=True
    )
