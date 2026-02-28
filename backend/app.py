import os
import time
import uuid
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, jsonify, request, abort, g, Response, stream_with_context
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
import requests
import json
from datetime import datetime, UTC, timedelta
import base64
import io
import math
import wave
import struct
import shutil
import subprocess
from pathlib import Path
import csv
import re
from urllib.parse import quote
from urllib.parse import unquote
from zipfile import ZipFile, ZIP_DEFLATED
from collections import deque
import importlib.util
from werkzeug.exceptions import HTTPException
import random
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas as rl_canvas
except ImportError:
    A4 = None
    ImageReader = None
    rl_canvas = None
try:
    import cairosvg
except ImportError:
    cairosvg = None
try:
    from apscheduler.schedulers.background import BackgroundScheduler
except ImportError:
    BackgroundScheduler = None
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseUpload
except ImportError:
    service_account = None
    build = None
    MediaIoBaseUpload = None
try:
    from pydantic import BaseModel, Field, ValidationError
except ImportError:
    BaseModel = None
    Field = None
    ValidationError = Exception
try:
    from openpyxl import Workbook
except Exception:
    Workbook = None
load_dotenv()

app = Flask(__name__)
default_db_path = os.path.join(os.path.dirname(__file__), 'instance', 'database.db')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', f'sqlite:///{default_db_path}')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['PROPAGATE_EXCEPTIONS'] = False

db = SQLAlchemy(app)

# Configuración de IA
OLLAMA_URL = os.getenv('OLLAMA_API_URL', 'http://localhost:11434')
PERPLEXITY_KEY = os.getenv('PERPLEXITY_API_KEY', '')
OPENAI_KEY = os.getenv('OPENAI_API_KEY', '')
GEMINI_KEY = os.getenv('GEMINI_API_KEY', '')
DEEPSEEK_KEY = os.getenv('DEEPSEEK_API_KEY', '')
GROQ_KEY = os.getenv('GROQ_API_KEY', '')
GLM_KEY = os.getenv('GLM_API_KEY', '')
QWEN_KEY = os.getenv('QWEN_API_KEY', '')
KIMI_KEY = os.getenv('KIMI_API_KEY', '')
KLING_KEY = os.getenv('KLING_API_KEY', '')
HUGGINGFACE_KEY = os.getenv('HUGGINGFACE_API_KEY', '')
MANUS_KEY = os.getenv('MANUS_API_KEY', '')

PERPLEXITY_MODEL = os.getenv('PERPLEXITY_MODEL', 'sonar')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')
GROQ_MODEL = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')
GLM_MODEL = os.getenv('GLM_MODEL', 'glm-4-flash')
QWEN_MODEL = os.getenv('QWEN_MODEL', 'qwen-plus')
QWEN_IMAGE_MODEL = os.getenv('QWEN_IMAGE_MODEL', 'qwen-image')
KIMI_MODEL = os.getenv('KIMI_MODEL', 'moonshot-v1-8k')
GEMINI_MODELS = [m.strip() for m in os.getenv('GEMINI_MODELS', 'gemini-2.0-flash,gemini-1.5-flash,gemini-1.5-pro').split(',') if m.strip()]
GEMINI_API_HOST = os.getenv('GEMINI_API_HOST', 'https://generativelanguage.googleapis.com')

GLM_BASE_URL = os.getenv('GLM_BASE_URL', 'https://open.bigmodel.cn/api/paas/v4')
QWEN_BASE_URL = os.getenv('QWEN_BASE_URL', 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1')
KIMI_BASE_URL = os.getenv('KIMI_BASE_URL', 'https://api.moonshot.cn/v1')
GROQ_BASE_URL = os.getenv('GROQ_BASE_URL', 'https://api.groq.com/openai/v1')
WAN_RUNTIME_URL = os.getenv('WAN_RUNTIME_URL', '')
WAN_API_KEY = os.getenv('WAN_API_KEY', '')
WAN_MODEL = os.getenv('WAN_MODEL', 'Wan2.1-T2V-1.3B')

AI_TEST_CACHE_TTL_SECONDS = int(os.getenv('AI_TEST_CACHE_TTL_SECONDS', '600'))
AI_TEST_CACHE = {
    'timestamp': None,
    'payload': None
}
BACKUP_HOUR = int(os.getenv('BACKUP_HOUR', '2'))
BACKUP_MINUTE = int(os.getenv('BACKUP_MINUTE', '30'))
BACKUP_RETENTION_COUNT = int(os.getenv('BACKUP_RETENTION_COUNT', '30'))
AUTO_RESTORE_FROM_BACKUP = os.getenv('AUTO_RESTORE_FROM_BACKUP', 'true').lower() == 'true'
FRANCAIS6_IMPORT_ROOT = os.getenv('FRANCAIS6_IMPORT_ROOT', '/Volumes/BEA/CURSO 2025-2026/FRANÇAIS 6º')
GOOGLE_SERVICE_ACCOUNT_FILE = os.getenv('GOOGLE_SERVICE_ACCOUNT_FILE', '')
GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON', '')
GOOGLE_DRIVE_ROOT_FOLDER_ID = os.getenv('GOOGLE_DRIVE_ROOT_FOLDER_ID', '')
GOOGLE_WORKSPACE_DEFAULT_CLASS = os.getenv('GOOGLE_WORKSPACE_DEFAULT_CLASS', '6º Primaria')
scheduler = None
MAGIC_ACTIVITY_TYPES = ['matching', 'color_match', 'dialogue', 'fill_blank', 'image_choice', 'label_image', 'scene_story']
MAGIC_ACTIVITY_INDEX = 0
QUALITY_MIN_SCORE = float(os.getenv('QUALITY_MIN_SCORE', '0.72'))
QUALITY_BATCH_MIN_SCORE = float(os.getenv('QUALITY_BATCH_MIN_SCORE', '0.76'))
DEFAULT_CHAT_SIMPLE_MODE = os.getenv('DEFAULT_CHAT_SIMPLE_MODE', 'true').lower() == 'true'
CANARIAS_COMPLIANCE_MODE = os.getenv('CANARIAS_COMPLIANCE_MODE', 'true').lower() == 'true'
AUDIT_LOG_MAX_ITEMS = int(os.getenv('AUDIT_LOG_MAX_ITEMS', '1000'))
AUDIT_LOG = deque(maxlen=AUDIT_LOG_MAX_ITEMS)
METRICS = {
    'audit_log_queries': 0,
    'total_audit_events': 0,
    'action_counts': {}
}
EXPORT_TOKEN_MAP = {}
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
PORT = int(os.getenv('PORT', '5012'))
HOST = os.getenv('HOST', '0.0.0.0')
DEBUG_MODE = os.getenv('DEBUG', 'false').lower() == 'true'
ALLOWED_ORIGINS = [origin.strip() for origin in os.getenv('ALLOWED_ORIGINS', 'http://localhost:5191,http://127.0.0.1:5191').split(',') if origin.strip()]
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv('RATE_LIMIT_WINDOW_SECONDS', '60'))
RATE_LIMIT_DEFAULT_REQUESTS = int(os.getenv('RATE_LIMIT_DEFAULT_REQUESTS', '30'))
RATE_LIMIT_STORE = {}
ENTERPRISE_FEATURES = {
    flag.strip()
    for flag in os.getenv(
        'ENTERPRISE_FEATURES',
        'batch_generation,batch_publish,semantic_search,ops_metrics,interactive_sessions'
    ).split(',')
    if flag.strip()
}

AI_KEY_REQUIREMENTS = {
    'perplexity': ['PERPLEXITY_API_KEY'],
    'openai': ['OPENAI_API_KEY'],
    'gemini': ['GEMINI_API_KEY'],
    'deepseek': ['DEEPSEEK_API_KEY'],
    'groq': ['GROQ_API_KEY'],
    'glm': ['GLM_API_KEY'],
    'qwen': ['QWEN_API_KEY'],
    'kimi': ['KIMI_API_KEY'],
    'kling': ['KLING_API_KEY'],
    'huggingface_ocr': ['HUGGINGFACE_API_KEY'],
    'manus': ['MANUS_API_KEY']
}

PROVIDER_COST_PERF = {
    # Lower cost score is better; higher performance score is better.
    'groq': {'cost_score': 1, 'performance_score': 4},
    'gemini': {'cost_score': 2, 'performance_score': 4},
    'perplexity': {'cost_score': 3, 'performance_score': 3},
    'openai': {'cost_score': 4, 'performance_score': 5},
}

PEDAGOGICAL_TEMPLATE_ENGINE = {
    'fill_blank': {
        'name': 'Completar contexto',
        'required_fields': ['question', 'correct_answer', 'options', 'hint', 'emoji'],
        'teacher_goal': 'Refuerzo de vocabulario con apoyo visual y pista breve'
    },
    'matching': {
        'name': 'Asociar conceptos',
        'required_fields': ['left', 'right', 'emoji'],
        'teacher_goal': 'Relacionar palabra-definición o francés-español'
    },
    'color_match': {
        'name': 'Colores y vocabulario',
        'required_fields': ['word', 'color_label', 'color_hex'],
        'teacher_goal': 'Conectar léxico con percepción visual'
    },
    'dialogue': {
        'name': 'Diálogo guiado',
        'required_fields': ['speaker', 'line_with_blank', 'options', 'correct_answer', 'emoji'],
        'teacher_goal': 'Practicar interacción oral/funcional'
    },
    'image_choice': {
        'name': 'Elección por imagen',
        'required_fields': ['question', 'choices', 'correct_answer', 'hint', 'emoji'],
        'teacher_goal': 'Comprensión semántica apoyada por imagen'
    },
    'label_image': {
        'name': 'Etiquetado visual',
        'required_fields': ['label', 'definition', 'image_url', 'emoji'],
        'teacher_goal': 'Nombrar y describir objetos con apoyo visual'
    },
    'scene_story': {
        'name': 'Secuencia narrativa',
        'required_fields': ['sentence', 'correct_order', 'image_url', 'emoji'],
        'teacher_goal': 'Orden temporal y producción narrativa'
    }
}

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), force=True)
app.logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
log_dir = Path(__file__).resolve().parent / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
file_handler = RotatingFileHandler(log_dir / 'app.log', maxBytes=5 * 1024 * 1024, backupCount=5, encoding='utf-8')
file_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
app.logger.addHandler(file_handler)
CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGINS}})

class Exercise(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    topic = db.Column(db.String(100), nullable=False)
    level = db.Column(db.String(10), nullable=False)
    exercise_type = db.Column(db.String(50), nullable=False)
    content = db.Column(db.JSON, nullable=False)
    ai_model = db.Column(db.String(50), default='local')
    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(UTC))

    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'topic': self.topic,
            'level': self.level,
            'exercise_type': self.exercise_type,
            'content': self.content,
            'image_url': generate_illustration_data_uri(self.topic, 'exercise'),
            'ai_model': self.ai_model,
            'created_at': self.created_at.isoformat()
        }

class Exam(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    exercises = db.Column(db.JSON, nullable=False)
    total_score = db.Column(db.Integer, default=100)
    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(UTC))

    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'exercises': self.exercises,
            'total_score': self.total_score,
            'image_url': generate_illustration_data_uri(self.title, 'exam'),
            'created_at': self.created_at.isoformat()
        }

class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    file_data = db.Column(db.LargeBinary)
    analysis = db.Column(db.JSON)
    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(UTC))

    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.filename,
            'analysis': self.analysis,
            'image_url': generate_illustration_data_uri(self.filename, 'document'),
            'created_at': self.created_at.isoformat()
        }


class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    role = db.Column(db.String(20), nullable=False)  # user|assistant
    content = db.Column(db.Text, nullable=False)
    task_type = db.Column(db.String(50), default='chat')
    provider = db.Column(db.String(50), default='llama3')
    model = db.Column(db.String(50), default='llama3')
    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(UTC))

    def to_dict(self):
        return {
            'id': self.id,
            'role': self.role,
            'content': self.content,
            'task_type': self.task_type,
            'provider': self.provider,
            'model': self.model,
            'created_at': self.created_at.isoformat()
        }


def audit_event(action, detail=None):
    AUDIT_LOG.appendleft({
        'timestamp': datetime.now(UTC).isoformat(),
        'action': action,
        'detail': detail or {}
    })
    # update metrics
    try:
        METRICS['total_audit_events'] = METRICS.get('total_audit_events', 0) + 1
        ac = METRICS.setdefault('action_counts', {})
        ac[action] = ac.get(action, 0) + 1
    except Exception:
        pass


def log_event(level, message, **fields):
    payload = {
        'timestamp': datetime.now(UTC).isoformat(),
        'message': message,
        **fields
    }
    app.logger.log(level, json.dumps(payload, ensure_ascii=False))


def api_error(status_code, code, message, detail=None):
    response = {
        'ok': False,
        'error': {
            'code': code,
            'message': message
        },
        'request_id': getattr(g, 'request_id', None)
    }
    if detail is not None:
        response['error']['detail'] = detail
    return jsonify(response), status_code


@app.before_request
def before_request():
    g.request_id = request.headers.get('X-Request-Id') or str(uuid.uuid4())
    g.request_start = time.perf_counter()
    log_event(
        logging.INFO,
        'request.start',
        request_id=g.request_id,
        method=request.method,
        path=request.path,
        remote_addr=request.headers.get('X-Forwarded-For', request.remote_addr)
    )
    limit = get_rate_limit_for_path(request.path)
    if limit is not None:
        ok, remaining = check_rate_limit(limit)
        if not ok:
            return api_error(429, 'rate_limited', 'Too many requests for this endpoint')
        g.rate_limit_remaining = remaining


@app.after_request
def after_request(response):
    response.headers['X-Request-Id'] = getattr(g, 'request_id', '')
    if hasattr(g, 'rate_limit_remaining'):
        response.headers['X-RateLimit-Remaining'] = str(g.rate_limit_remaining)
    duration_ms = None
    if hasattr(g, 'request_start'):
        duration_ms = round((time.perf_counter() - g.request_start) * 1000, 2)
    log_event(
        logging.INFO,
        'request.end',
        request_id=getattr(g, 'request_id', None),
        method=request.method,
        path=request.path,
        status_code=response.status_code,
        duration_ms=duration_ms
    )
    return response


@app.errorhandler(HTTPException)
def handle_http_exception(err):
    log_event(
        logging.WARNING,
        'request.http_error',
        request_id=getattr(g, 'request_id', None),
        method=request.method,
        path=request.path,
        status_code=err.code,
        detail=err.description
    )
    return api_error(err.code or 500, f'http_{err.code or 500}', err.description or 'HTTP error')


@app.errorhandler(Exception)
def handle_exception(err):
    log_event(
        logging.ERROR,
        'request.unhandled_exception',
        request_id=getattr(g, 'request_id', None),
        method=request.method,
        path=request.path,
        error_type=type(err).__name__,
        detail=str(err)
    )
    return api_error(500, 'internal_server_error', 'Unexpected server error')


def get_rate_limit_for_path(path):
    sensitive_limits = {
        '/api/exercises/generate': 20,
        '/api/exercises/generate-batch': 8,
        '/api/ai/test': 10,
        '/api/ai/providers/': 20,
        '/api/backups/export': 10,
        '/api/backups/restore-latest': 5,
        '/api/library/repair-exercises': 5,
        '/api/library/import-francais6': 3,
        '/api/exercises/repair-batch': 5,
        '/api/google/workspace/publish': 20,
        '/api/google/workspace/publish-batch': 8
    }
    for key, value in sensitive_limits.items():
        if key.endswith('/') and path.startswith(key):
            return value
        if path == key:
            return value
    if path.startswith('/api/'):
        return RATE_LIMIT_DEFAULT_REQUESTS
    return None


def check_rate_limit(limit):
    ip = request.headers.get('X-Forwarded-For', request.remote_addr) or 'unknown'
    key = f'{ip}:{request.path}'
    now = time.time()
    start, count = RATE_LIMIT_STORE.get(key, (now, 0))
    if now - start >= RATE_LIMIT_WINDOW_SECONDS:
        start, count = now, 0
    count += 1
    RATE_LIMIT_STORE[key] = (start, count)
    remaining = max(0, limit - count)
    return count <= limit, remaining


def feature_enabled(flag):
    return flag in ENTERPRISE_FEATURES


def module_installed(module_name):
    return importlib.util.find_spec(module_name) is not None


def as_utc(dt_value):
    if dt_value is None:
        return None
    if dt_value.tzinfo is None:
        return dt_value.replace(tzinfo=UTC)
    return dt_value.astimezone(UTC)


def sanitize_for_cloud_prompt(text):
    if not isinstance(text, str):
        return text, []
    detections = []
    sanitized = text
    pii_patterns = [
        ('email', r'[\w\.-]+@[\w\.-]+\.\w+'),
        ('phone', r'(\+?\d[\d\-\s]{7,}\d)'),
        ('dni_like', r'\b\d{8}[A-Za-z]\b')
    ]
    try:
        import re
        for name, pattern in pii_patterns:
            if re.search(pattern, sanitized):
                detections.append(name)
                sanitized = re.sub(pattern, '[REDACTED]', sanitized)
    except Exception:
        pass
    return sanitized, detections

def call_ollama(prompt, model='llama3'):
    try:
        response = requests.post(
            f'{OLLAMA_URL}/api/generate',
            json={'model': model, 'prompt': prompt, 'stream': False},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()['response']
    except Exception as e:
        log_event(logging.WARNING, 'provider.ollama.error', detail=str(e))
    return None

def call_openai_compatible(prompt, *, provider_name, api_key, model, base_url):
    try:
        if not api_key:
            return None
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        data = {
            'model': model,
            'messages': [{'role': 'user', 'content': prompt}]
        }
        response = requests.post(f'{base_url}/chat/completions', json=data, headers=headers, timeout=30)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        log_event(logging.WARNING, f'provider.{provider_name}.http_status', status_code=response.status_code)
    except Exception as e:
        log_event(logging.WARNING, f'provider.{provider_name}.error', detail=str(e))
    return None

def call_perplexity(prompt):
    try:
        if not PERPLEXITY_KEY:
            return None
        headers = {
            'Authorization': f'Bearer {PERPLEXITY_KEY}',
            'Content-Type': 'application/json'
        }
        data = {'model': PERPLEXITY_MODEL, 'messages': [{'role': 'user', 'content': prompt}]}
        response = requests.post('https://api.perplexity.ai/chat/completions', json=data, headers=headers, timeout=30)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        log_event(logging.WARNING, 'provider.perplexity.http_status', status_code=response.status_code)
    except Exception as e:
        log_event(logging.WARNING, 'provider.perplexity.error', detail=str(e))
    return None


def call_openai(prompt):
    return call_openai_compatible(
        prompt,
        provider_name='openai',
        api_key=OPENAI_KEY,
        model=OPENAI_MODEL,
        base_url='https://api.openai.com/v1'
    )


def call_gemini_generate(prompt, timeout=30):
    if not GEMINI_KEY:
        return None, 'missing_api_key:GEMINI_API_KEY'
    data = {'contents': [{'parts': [{'text': prompt}]}]}
    versions = ('v1beta', 'v1')
    last_detail = 'no_response'
    for version in versions:
        for model_name in GEMINI_MODELS:
            try:
                url = f'{GEMINI_API_HOST}/{version}/models/{model_name}:generateContent'
                response = requests.post(
                    url,
                    json=data,
                    headers={'x-goog-api-key': GEMINI_KEY, 'Content-Type': 'application/json'},
                    timeout=timeout
                )
                if response.status_code == 200:
                    try:
                        return response.json()['candidates'][0]['content']['parts'][0]['text'], 'reachable'
                    except Exception:
                        return None, 'invalid_payload'
                if response.status_code == 404:
                    last_detail = f'model_not_found:{model_name}'
                    continue
                if response.status_code == 429:
                    return None, 'rate_limit'
                if response.status_code in (401, 403):
                    return None, 'auth_error'
                last_detail = f'http_{response.status_code}'
            except Exception as e:
                last_detail = f'network_error:{str(e)}'
    return None, last_detail


def detect_best_gemini_models(timeout=10):
    candidates = []
    for model in GEMINI_MODELS + ['gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-1.5-pro']:
        if model and model not in candidates:
            candidates.append(model)

    tested = []
    reachable = []
    for model in candidates:
        start = time.perf_counter()
        text, detail = call_gemini_generate(f'Responde solo: OK ({model})', timeout=timeout)
        latency_ms = int((time.perf_counter() - start) * 1000)
        tested.append({'model': model, 'detail': detail, 'latency_ms': latency_ms})
        if text and detail == 'reachable':
            reachable.append({'model': model, 'latency_ms': latency_ms})

    reachable.sort(key=lambda item: item['latency_ms'])
    recommended_models = [item['model'] for item in reachable] or GEMINI_MODELS
    recommended_env_value = ','.join(recommended_models)
    return {
        'tested': tested,
        'reachable': reachable,
        'recommended_models': recommended_models,
        'recommended_env_value': recommended_env_value
    }


def recommend_fallback_provider():
    provider_checks = {}
    candidates = ['groq', 'gemini', 'perplexity', 'openai']
    for provider in candidates:
        ok, detail = test_provider(provider)
        provider_checks[provider] = {'ok': ok, 'detail': detail}

    available = [provider for provider in candidates if provider_checks[provider]['ok']]
    if not available:
        return {
            'provider': 'llama3',
            'reason': 'No cloud providers currently reachable; fallback to local llama3',
            'provider_checks': provider_checks
        }

    ranked = sorted(
        available,
        key=lambda p: (
            PROVIDER_COST_PERF.get(p, {}).get('cost_score', 99),
            -PROVIDER_COST_PERF.get(p, {}).get('performance_score', 0)
        )
    )
    selected = ranked[0]
    meta = PROVIDER_COST_PERF.get(selected, {'cost_score': 99, 'performance_score': 0})
    return {
        'provider': selected,
        'reason': f"Selected by best cost/performance score (cost={meta['cost_score']}, perf={meta['performance_score']})",
        'provider_checks': provider_checks
    }


def call_gemini(prompt):
    text, detail = call_gemini_generate(prompt, timeout=30)
    if text:
        return text
    log_event(logging.WARNING, 'provider.gemini.error', detail=detail)
    return None


def call_deepseek(prompt):
    try:
        if not DEEPSEEK_KEY:
            return None
        headers = {
            'Authorization': f'Bearer {DEEPSEEK_KEY}',
            'Content-Type': 'application/json'
        }
        data = {
            'model': DEEPSEEK_MODEL,
            'messages': [{'role': 'user', 'content': prompt}]
        }
        response = requests.post('https://api.deepseek.com/chat/completions', json=data, headers=headers, timeout=30)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        if response.status_code == 402:
            log_event(logging.WARNING, 'provider.deepseek.http_status', status_code=402, detail='insufficient_credits')
            return None
        log_event(logging.WARNING, 'provider.deepseek.http_status', status_code=response.status_code)
    except Exception as e:
        log_event(logging.WARNING, 'provider.deepseek.error', detail=str(e))
    return None


def call_glm(prompt):
    return call_openai_compatible(
        prompt,
        provider_name='glm',
        api_key=GLM_KEY,
        model=GLM_MODEL,
        base_url=GLM_BASE_URL
    )


def call_qwen(prompt):
    return call_openai_compatible(
        prompt,
        provider_name='qwen',
        api_key=QWEN_KEY,
        model=QWEN_MODEL,
        base_url=QWEN_BASE_URL
    )


def call_kimi(prompt):
    return call_openai_compatible(
        prompt,
        provider_name='kimi',
        api_key=KIMI_KEY,
        model=KIMI_MODEL,
        base_url=KIMI_BASE_URL
    )


def call_groq(prompt):
    return call_openai_compatible(
        prompt,
        provider_name='groq',
        api_key=GROQ_KEY,
        model=GROQ_MODEL,
        base_url=GROQ_BASE_URL
    )


def get_or_404(model, entity_id, name):
    instance = db.session.get(model, entity_id)
    if instance is None:
        abort(404, description=f'{name} not found')
    return instance


def generate_illustration_data_uri(text, kind='exercise'):
    palette = {
        'exercise': ('#1d4ed8', '#0ea5e9'),
        'exam': ('#7c2d12', '#f97316'),
        'document': ('#166534', '#22c55e')
    }
    start, end = palette.get(kind, ('#334155', '#64748b'))
    safe_text = (text or kind).strip()[:40]
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1200 630'>
<defs><linearGradient id='g' x1='0' y1='0' x2='1' y2='1'>
<stop offset='0%' stop-color='{start}'/><stop offset='100%' stop-color='{end}'/></linearGradient></defs>
<rect width='1200' height='630' fill='url(#g)'/>
<rect x='48' y='48' width='1104' height='534' rx='24' fill='rgba(255,255,255,0.12)'/>
<text x='80' y='180' fill='white' font-size='44' font-family='Arial, sans-serif'>French Exercise App</text>
<text x='80' y='260' fill='white' font-size='68' font-family='Arial, sans-serif'>{safe_text}</text>
<text x='80' y='330' fill='white' font-size='34' font-family='Arial, sans-serif'>Type: {kind}</text>
</svg>"""
    return f"data:image/svg+xml;utf8,{quote(svg)}"


def ensure_image_url(value, kind='exercise'):
    if isinstance(value, str) and value.startswith('data:image'):
        return value
    return generate_illustration_data_uri(value or kind, kind)


def generate_with_provider(prompt, ai_mode='local', ai_model='llama3'):
    detections = []
    if ai_mode != 'local' and CANARIAS_COMPLIANCE_MODE:
        prompt, detections = sanitize_for_cloud_prompt(prompt)
    if ai_mode == 'local':
        local_model = 'mistral' if ai_model == 'mistral' else 'llama3'
        if detections:
            audit_event('compliance.sanitize_prompt', {'provider': ai_model, 'detections': detections})
        return call_ollama(prompt, model=local_model)

    cloud_dispatch = {
        'perplexity': call_perplexity,
        'openai': call_openai,
        'gemini': call_gemini,
        'deepseek': call_deepseek,
        'groq': call_groq,
        'glm': call_glm,
        'qwen': call_qwen,
        'kimi': call_kimi
    }
    provider = ai_model if ai_model in cloud_dispatch else 'perplexity'
    if detections:
        audit_event('compliance.sanitize_prompt', {'provider': provider, 'detections': detections})
    audit_event('ai.generate.request', {'provider': provider, 'mode': ai_mode})
    result = cloud_dispatch[provider](prompt)
    if result:
        return result

    # DeepSeek often fails with 402 credits; auto-fallback keeps UX operational.
    if provider == 'deepseek':
        fallback_order = ['groq', 'openai', 'perplexity', 'gemini']
        for fallback_provider in fallback_order:
            fn = cloud_dispatch.get(fallback_provider)
            if not fn:
                continue
            fallback_result = fn(prompt)
            if fallback_result:
                audit_event('ai.generate.fallback', {'from': 'deepseek', 'to': fallback_provider})
                return fallback_result
    return None


def generate_with_provider_stream(prompt, ai_mode='local', ai_model='llama3'):
    detections = []
    if ai_mode != 'local' and CANARIAS_COMPLIANCE_MODE:
        prompt, detections = sanitize_for_cloud_prompt(prompt)

    if ai_mode == 'local':
        local_model = 'mistral' if ai_model == 'mistral' else 'llama3'
        if detections:
            audit_event('compliance.sanitize_prompt', {'provider': ai_model, 'detections': detections})
        try:
            response = requests.post(
                f'{OLLAMA_URL}/api/generate',
                json={'model': local_model, 'prompt': prompt, 'stream': True},
                timeout=90,
                stream=True
            )
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    payload = json.loads(line.decode('utf-8'))
                except Exception:
                    continue
                token = payload.get('response', '')
                if token:
                    yield token
                if payload.get('done'):
                    break
            return
        except Exception:
            pass

    fallback = generate_with_provider(prompt, ai_mode=ai_mode, ai_model=ai_model)
    if not fallback:
        return
    for part in str(fallback).split():
        yield part + ' '


def get_backup_paths():
    backend_dir = Path(__file__).resolve().parent
    db_path = Path(app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', ''))
    backup_dir = backend_dir / 'backups'
    backup_dir.mkdir(parents=True, exist_ok=True)
    return db_path, backup_dir


def get_exports_dir():
    backend_dir = Path(__file__).resolve().parent
    exports_dir = backend_dir / 'exports'
    exports_dir.mkdir(parents=True, exist_ok=True)
    return exports_dir


def export_backup():
    db_path, backup_dir = get_backup_paths()
    timestamp = datetime.now(UTC).strftime('%Y%m%d_%H%M%S')
    backup_prefix = backup_dir / f'backup_{timestamp}'
    sqlite_backup = backup_prefix.with_suffix('.db')
    json_backup = backup_prefix.with_suffix('.json')

    exercises = [e.to_dict() for e in Exercise.query.all()]
    exams = [e.to_dict() for e in Exam.query.all()]
    documents = [d.to_dict() for d in Document.query.all()]
    payload = {
        'generated_at': datetime.now(UTC).isoformat(),
        'counts': {
            'exercises': len(exercises),
            'exams': len(exams),
            'documents': len(documents)
        },
        'data': {
            'exercises': exercises,
            'exams': exams,
            'documents': documents
        }
    }

    with open(json_backup, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    if db_path.exists():
        shutil.copy2(db_path, sqlite_backup)

    rotate_backups(backup_dir, BACKUP_RETENTION_COUNT)

    return {
        'sqlite_backup': str(sqlite_backup),
        'json_backup': str(json_backup),
        'generated_at': payload['generated_at'],
        'counts': payload['counts']
    }


def rotate_backups(backup_dir, keep_count=30):
    json_files = sorted(backup_dir.glob('backup_*.json'))
    sqlite_files = sorted(backup_dir.glob('backup_*.db'))

    # Keep newest N by filename timestamp; remove older files.
    old_json = json_files[:-keep_count] if len(json_files) > keep_count else []
    old_sqlite = sqlite_files[:-keep_count] if len(sqlite_files) > keep_count else []

    for file_path in old_json + old_sqlite:
        try:
            file_path.unlink(missing_ok=True)
        except Exception as e:
            log_event(logging.WARNING, 'backup.rotation.delete_failed', file=str(file_path), detail=str(e))


def sanitize_filename(value):
    safe = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in (value or 'item'))
    return safe[:80].strip('_') or 'item'


def build_library_item(item_type, model):
    data = model.to_dict()
    data['item_type'] = item_type
    data['display_title'] = data.get('title') or data.get('filename') or f'{item_type}-{data.get("id")}'
    return data


def infer_template_topic_and_type(path_obj):
    value = str(path_obj).lower()
    name = path_obj.stem.lower()

    topic = 'français'
    if any(k in value for k in ['vetement', 'vêtement', 'armoire', 'porte']):
        topic = 'les vêtements'
    elif any(k in value for k in ['heure', 'horloge', 'emploi du temps', 'nombres']):
        topic = 'les heures'
    elif any(k in value for k in ['corps', 'avoir mal', 'description']):
        topic = 'le corps'
    elif any(k in value for k in ['france', 'tout sur moi', 'premièrs jours']):
        topic = 'la france'

    activity_type = 'matching'
    if any(k in name for k in ['colorie', 'couleur']):
        activity_type = 'color_match'
    elif any(k in name for k in ['dict', 'question', 'oral']):
        activity_type = 'dialogue'
    elif any(k in name for k in ['complète', 'complete', 'crucigrama', 'mots mêlés', 'mots mêlés']):
        activity_type = 'fill_blank'

    return topic, activity_type


def build_imported_template_content(source_path, topic, activity_type, title):
    content = fallback_creative_content(topic, activity_type)
    content['title'] = title
    content['template_prompt'] = (
        f"Crear actividad escolar en francés basada en material docente existente. "
        f"Tema: {topic}. Tipo sugerido: {activity_type}. "
        f"Adaptar para 6º primaria y mantener formato imprimible + interactivo."
    )
    content['import_metadata'] = {
        'source_path': str(source_path),
        'source_file': Path(source_path).name,
        'source_format': Path(source_path).suffix.lower().lstrip('.'),
        'import_mode': 'francais6_auto_template',
        'imported_at': datetime.now(UTC).isoformat()
    }
    content['image_url'] = generate_illustration_data_uri(topic, 'exercise')
    return content


def simple_pdf_bytes(title, lines):
    # Minimal single-page PDF writer (ASCII-safe text only).
    def esc(s):
        return s.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')

    content_lines = [f"BT /F1 12 Tf 50 790 Td ({esc(title)}) Tj ET"]
    y = 770
    for line in lines[:50]:
        safe_line = ''.join(ch if ord(ch) < 128 else '?' for ch in line)
        content_lines.append(f"BT /F1 10 Tf 50 {y} Td ({esc(safe_line)}) Tj ET")
        y -= 14
        if y < 60:
            break
    stream = '\n'.join(content_lines).encode('latin-1', errors='replace')

    objects = []
    objects.append(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n")
    objects.append(b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n")
    objects.append(b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n")
    objects.append(b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n")
    objects.append(b"5 0 obj << /Length " + str(len(stream)).encode() + b" >> stream\n" + stream + b"\nendstream endobj\n")

    pdf = b"%PDF-1.4\n"
    xref_positions = [0]
    for obj in objects:
        xref_positions.append(len(pdf))
        pdf += obj
    xref_start = len(pdf)
    pdf += f"xref\n0 {len(xref_positions)}\n".encode()
    pdf += b"0000000000 65535 f \n"
    for pos in xref_positions[1:]:
        pdf += f"{pos:010} 00000 n \n".encode()
    pdf += f"trailer << /Size {len(xref_positions)} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF\n".encode()
    return pdf


def build_worksheet_lines(item_type, item_data, include_answers=True, worksheet_role='teacher'):
    lines = []
    title = item_data.get('display_title') or item_data.get('title') or f'Ficha {item_type}'
    lines.append(f'Titulo: {title}')
    if item_data.get('topic'):
        lines.append(f'Tema: {item_data.get("topic")}')
    if item_data.get('level'):
        lines.append(f'Nivel: {item_data.get("level")}')
    lines.append(f'Version: {"Profesor" if worksheet_role == "teacher" else "Alumno"}')
    lines.append('Nombre del alumno: ______________________   Fecha: ______________')
    lines.append('')

    def append_activity(content, prefix=''):
        items = content.get('items') if isinstance(content.get('items'), list) else []
        activity_type = content.get('activity_type', 'fill_blank')
        lines.append(f'{prefix}Actividad: {activity_type}')
        lines.append('Instrucciones: Lee y responde en los espacios.')
        lines.append('')
        for idx, row in enumerate(items, start=1):
            if activity_type == 'matching':
                left = row.get('left', '')
                lines.append(f'{idx}. {left}  ->  __________________________')
                if include_answers:
                    lines.append(f'   Solucion: {row.get("right", "")}')
            elif activity_type == 'color_match':
                word = row.get('word', '')
                lines.append(f'{idx}. {word}  ->  color: _____________________')
                if include_answers:
                    lines.append(f'   Solucion: {row.get("color_label", "")}')
            elif activity_type == 'dialogue':
                line_blank = row.get('line_with_blank', '')
                lines.append(f'{idx}. {line_blank}')
                opts = row.get('options') if isinstance(row.get('options'), list) else []
                if opts:
                    lines.append(f'   Opciones: {" | ".join(str(opt) for opt in opts)}')
                if include_answers:
                    lines.append(f'   Solucion: {row.get("correct_answer", "")}')
            elif activity_type == 'image_choice':
                question = row.get('question', '')
                lines.append(f'{idx}. {question}')
                options = row.get('options') if isinstance(row.get('options'), list) else []
                if not options and isinstance(row.get('choices'), list):
                    options = row.get('choices')
                for letter, option in zip(['A', 'B', 'C', 'D'], options[:4]):
                    label = option.get('label', '') if isinstance(option, dict) else str(option)
                    lines.append(f'   {letter}) {label}')
                lines.append('   Respuesta: ______')
                if include_answers:
                    lines.append(f'   Solucion: {row.get("correct_answer", "")}')
            elif activity_type == 'label_image':
                label = row.get('label', '')
                lines.append(f'{idx}. {label}: ______________________________')
                if include_answers:
                    lines.append(f'   Solucion: {row.get("definition", "")}')
            elif activity_type == 'scene_story':
                sentence = row.get('sentence', '')
                lines.append(f'{idx}. {sentence}')
                lines.append('   Orden: ______')
                if include_answers:
                    lines.append(f'   Solucion: {row.get("correct_order", idx)}')
            else:
                question = row.get('question', '')
                lines.append(f'{idx}. {question}')
                lines.append('   Respuesta: __________________________')
                if include_answers:
                    lines.append(f'   Solucion: {row.get("correct_answer", "")}')
            lines.append('')

    if item_type == 'exercise':
        content = item_data.get('content') if isinstance(item_data.get('content'), dict) else {}
        append_activity(content)
    elif item_type == 'exam':
        exam_exercises = item_data.get('exercises') if isinstance(item_data.get('exercises'), list) else []
        lines.append(f'Examen con {len(exam_exercises)} ejercicios')
        lines.append('')
        for i, ex in enumerate(exam_exercises, start=1):
            ex_content = ex.get('content') if isinstance(ex.get('content'), dict) else {}
            ex_title = ex.get('title') or f'Ejercicio {i}'
            lines.append(f'===== {ex_title} =====')
            append_activity(ex_content, prefix=f'[{i}] ')
    else:
        lines.append('Vista documental:')
        lines.extend(json.dumps(item_data, ensure_ascii=False, indent=2).splitlines())
    return lines


def extract_image_bytes_for_pdf(image_url):
    if not isinstance(image_url, str) or not image_url:
        return None

    def svg_to_png(svg_bytes):
        if cairosvg is None:
            return None
        try:
            return cairosvg.svg2png(bytestring=svg_bytes)
        except Exception as exc:
            log_event(logging.WARNING, 'pdf.svg_to_png_failed', detail=str(exc))
            return None

    try:
        if image_url.startswith('data:image/png;base64,'):
            return base64.b64decode(image_url.split(',', 1)[1])
        if image_url.startswith('data:image/jpeg;base64,'):
            return base64.b64decode(image_url.split(',', 1)[1])
        if image_url.startswith('data:image/svg+xml'):
            svg_payload = image_url.split(',', 1)[1] if ',' in image_url else ''
            if ';base64' in image_url:
                svg_bytes = base64.b64decode(svg_payload)
            else:
                svg_bytes = unquote(svg_payload).encode('utf-8')
            return svg_to_png(svg_bytes)
        if image_url.startswith('https://') or image_url.startswith('http://'):
            response = requests.get(image_url, timeout=10)
            if response.status_code == 200:
                content_type = (response.headers.get('content-type') or '').lower()
                if 'image/svg+xml' in content_type or image_url.lower().endswith('.svg'):
                    return svg_to_png(response.content)
                return response.content
    except Exception:
        return None
    return None


def worksheet_pdf_bytes(item_type, item_data, include_answers=True, worksheet_role='teacher'):
    lines = build_worksheet_lines(
        item_type,
        item_data,
        include_answers=include_answers,
        worksheet_role=worksheet_role,
    )
    title = item_data.get('display_title') or item_data.get('title') or f'Ficha {item_type}'
    image_url = item_data.get('image_url') or ((item_data.get('content') or {}).get('image_url') if isinstance(item_data.get('content'), dict) else '')

    if rl_canvas is None or A4 is None:
        return simple_pdf_bytes(title, lines)

    buffer = io.BytesIO()
    c = rl_canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    def draw_header():
        c.setFont('Helvetica-Bold', 16)
        c.drawString(42, height - 50, title[:80])
        c.setFont('Helvetica', 10)
        c.drawString(42, height - 68, f'Tipo: {item_type}')

    draw_header()
    y = height - 90

    img_bytes = extract_image_bytes_for_pdf(image_url)
    if img_bytes and ImageReader:
        try:
            reader = ImageReader(io.BytesIO(img_bytes))
            c.drawImage(reader, 42, y - 140, width=150, height=120, preserveAspectRatio=True, mask='auto')
            y -= 150
        except Exception:
            pass

    c.setFont('Helvetica', 11)
    for raw in lines:
        line = ''.join(ch if ord(ch) < 128 else '?' for ch in str(raw))
        if y < 48:
            c.showPage()
            draw_header()
            y = height - 90
            c.setFont('Helvetica', 11)
        c.drawString(42, y, line[:115])
        y -= 16

    c.showPage()
    c.save()
    return buffer.getvalue()


def build_google_workspace_html(item_type, item_data):
    pretty_json = json.dumps(item_data, ensure_ascii=False, indent=2)
    content = item_data.get('content') if isinstance(item_data.get('content'), dict) else {}
    title = item_data.get('display_title') or f'{item_type} export'
    hero_image = item_data.get('image_url') or content.get('image_url') or ''
    activity_type = content.get('activity_type', '')
    items = content.get('items') if isinstance(content.get('items'), list) else []

    rows = []
    for idx, item in enumerate(items[:20], start=1):
        if activity_type == 'matching':
            left = item.get('left', '')
            right = item.get('right', '')
            rows.append(f"<tr><td>{idx}</td><td>{left}</td><td>{right}</td></tr>")
        elif activity_type == 'color_match':
            word = item.get('word', '')
            label = item.get('color_label', '')
            rows.append(f"<tr><td>{idx}</td><td>{word}</td><td>{label}</td></tr>")
        elif activity_type == 'dialogue':
            speaker = item.get('speaker', f'Personaje {idx}')
            line = item.get('line_with_blank', '')
            answer = item.get('correct_answer', '')
            rows.append(f"<tr><td>{idx}</td><td><strong>{speaker}</strong>: {line}</td><td>{answer}</td></tr>")
        elif activity_type == 'image_choice':
            question = item.get('question', '')
            answer = item.get('correct_answer', '')
            rows.append(f"<tr><td>{idx}</td><td>{question}</td><td>{answer}</td></tr>")
        elif activity_type == 'label_image':
            label = item.get('label', '')
            definition = item.get('definition', '')
            rows.append(f"<tr><td>{idx}</td><td>{label}</td><td>{definition}</td></tr>")
        elif activity_type == 'scene_story':
            sentence = item.get('sentence', '')
            correct_order = item.get('correct_order', idx)
            rows.append(f"<tr><td>{idx}</td><td>{sentence}</td><td>{correct_order}</td></tr>")
        else:
            question = item.get('question', '')
            answer = item.get('correct_answer', '')
            rows.append(f"<tr><td>{idx}</td><td>{question}</td><td>{answer}</td></tr>")

    table_html = ''.join(rows) or "<tr><td colspan='3'>Sin filas estructuradas</td></tr>"
    image_block = f"<img src='{hero_image}' alt='preview' style='max-width:100%;border-radius:12px;border:1px solid #d0d7e2;'/>" if hero_image else ''
    return f"""<!doctype html>
<html>
<head>
  <meta charset='utf-8'/>
  <title>{title}</title>
  <meta name='viewport' content='width=device-width, initial-scale=1'/>
  <style>
    body {{ font-family: 'Avenir Next', Arial, sans-serif; margin: 24px; color:#172131; }}
    h1 {{ color:#0b3d91; margin-bottom:8px; }}
    .meta {{ color:#4a5568; margin-bottom: 14px; }}
    .card {{ border:1px solid #d8e0ec; border-radius:14px; padding:14px; margin-bottom:16px; background:#ffffff; }}
    table {{ width:100%; border-collapse: collapse; }}
    th,td {{ border:1px solid #d8e0ec; padding:8px; text-align:left; vertical-align:top; }}
    th {{ background:#f5f8ff; }}
    .json {{ white-space: pre-wrap; background:#f7f9ff; border:1px solid #dce5f3; border-radius:10px; padding:10px; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class='meta'>Tipo: {item_type} · Actividad: {activity_type or 'n/a'}</div>
  <div class='card'>{image_block}</div>
  <div class='card'>
    <table>
      <thead><tr><th>#</th><th>Contenido</th><th>Solución / Meta</th></tr></thead>
      <tbody>{table_html}</tbody>
    </table>
  </div>
  <h2>JSON fuente</h2>
  <div class='json'>{pretty_json}</div>
</body>
</html>"""


def google_workspace_ready():
    if service_account is None or build is None or MediaIoBaseUpload is None:
        return False, 'google_api_dependencies_missing'
    if not GOOGLE_SERVICE_ACCOUNT_FILE and not GOOGLE_SERVICE_ACCOUNT_JSON:
        return False, 'missing_google_service_account'
    return True, 'ready'


def load_google_service_account_info():
    if GOOGLE_SERVICE_ACCOUNT_JSON:
        try:
            return json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
        except Exception as exc:
            raise ValueError(f'invalid_GOOGLE_SERVICE_ACCOUNT_JSON:{exc}') from exc
    if GOOGLE_SERVICE_ACCOUNT_FILE:
        path = Path(GOOGLE_SERVICE_ACCOUNT_FILE)
        if not path.exists():
            raise ValueError('google_service_account_file_not_found')
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except Exception as exc:
            raise ValueError(f'invalid_service_account_file:{exc}') from exc
    raise ValueError('missing_google_service_account')


def get_google_workspace_clients():
    info = load_google_service_account_info()
    scopes = [
        'https://www.googleapis.com/auth/drive',
        'https://www.googleapis.com/auth/documents'
    ]
    creds = service_account.Credentials.from_service_account_info(info, scopes=scopes)
    drive_service = build('drive', 'v3', credentials=creds, cache_discovery=False)
    docs_service = build('docs', 'v1', credentials=creds, cache_discovery=False)
    return drive_service, docs_service


def drive_find_folder(drive_service, name, parent_id=None):
    escaped_name = str(name).replace("'", "\\'")
    query = f"mimeType='application/vnd.google-apps.folder' and trashed=false and name='{escaped_name}'"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    response = drive_service.files().list(
        q=query,
        pageSize=1,
        fields='files(id,name,webViewLink)'
    ).execute()
    files = response.get('files') or []
    return files[0] if files else None


def drive_create_folder(drive_service, name, parent_id=None):
    payload = {
        'name': name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    if parent_id:
        payload['parents'] = [parent_id]
    created = drive_service.files().create(body=payload, fields='id,name,webViewLink').execute()
    return created


def drive_ensure_folder(drive_service, name, parent_id=None):
    found = drive_find_folder(drive_service, name, parent_id=parent_id)
    if found:
        return found, False
    created = drive_create_folder(drive_service, name, parent_id=parent_id)
    return created, True


def publish_item_to_google_workspace(item_type, item_data, class_name):
    ready, reason = google_workspace_ready()
    if not ready:
        raise ValueError(reason)

    drive_service, _docs_service = get_google_workspace_clients()
    safe_class = (class_name or GOOGLE_WORKSPACE_DEFAULT_CLASS or 'Clase').strip()
    root_parent = GOOGLE_DRIVE_ROOT_FOLDER_ID.strip() or None

    class_folder, folder_created = drive_ensure_folder(drive_service, safe_class, parent_id=root_parent)
    html = build_google_workspace_html(item_type, item_data)
    filename = sanitize_filename(f"{item_data.get('display_title', item_type)}_{safe_class}.html")
    media = MediaIoBaseUpload(io.BytesIO(html.encode('utf-8')), mimetype='text/html', resumable=False)
    metadata = {
        'name': filename,
        'mimeType': 'application/vnd.google-apps.document',
        'parents': [class_folder['id']]
    }
    created_doc = drive_service.files().create(
        body=metadata,
        media_body=media,
        fields='id,name,webViewLink,parents'
    ).execute()
    return {
        'folder_id': class_folder.get('id'),
        'folder_name': class_folder.get('name'),
        'folder_url': class_folder.get('webViewLink') or f"https://drive.google.com/drive/folders/{class_folder.get('id')}",
        'folder_created': folder_created,
        'doc_id': created_doc.get('id'),
        'doc_name': created_doc.get('name'),
        'doc_url': created_doc.get('webViewLink') or f"https://docs.google.com/document/d/{created_doc.get('id')}/edit",
        'class_name': safe_class
    }


def export_item_file(item_type, item_data, export_format, options=None):
    options = options or {}
    exports_dir = get_exports_dir()
    timestamp = datetime.now(UTC).strftime('%Y%m%d_%H%M%S')
    base_name = sanitize_filename(f"{item_type}_{item_data.get('display_title')}_{timestamp}")

    if export_format == 'json':
        out = exports_dir / f'{base_name}.json'
        out.write_text(json.dumps(item_data, ensure_ascii=False, indent=2), encoding='utf-8')
        return out

    if export_format == 'txt':
        lines = [
            f"Tipo: {item_type}",
            f"Título: {item_data.get('display_title')}",
            f"ID: {item_data.get('id')}",
            '',
            json.dumps(item_data, ensure_ascii=False, indent=2)
        ]
        out = exports_dir / f'{base_name}.txt'
        out.write_text('\n'.join(lines), encoding='utf-8')
        return out

    if export_format == 'html':
        body = f"<h1>{item_data.get('display_title')}</h1><pre>{json.dumps(item_data, ensure_ascii=False, indent=2)}</pre>"
        html = f"<!doctype html><html><head><meta charset='utf-8'><title>{item_data.get('display_title')}</title></head><body>{body}</body></html>"
        out = exports_dir / f'{base_name}.html'
        out.write_text(html, encoding='utf-8')
        return out

    if export_format == 'google_workspace':
        html = build_google_workspace_html(item_type, item_data)
        out = exports_dir / f'{base_name}.gworkspace.html'
        out.write_text(html, encoding='utf-8')
        return out

    if export_format == 'pdf':
        out = exports_dir / f'{base_name}.pdf'
        include_answers = bool(options.get('include_answers', True))
        worksheet_role = str(options.get('worksheet_role') or ('teacher' if include_answers else 'student')).strip().lower()
        if worksheet_role not in {'teacher', 'student'}:
            worksheet_role = 'teacher' if include_answers else 'student'
        out.write_bytes(
            worksheet_pdf_bytes(
                item_type,
                item_data,
                include_answers=include_answers,
                worksheet_role=worksheet_role,
            )
        )
        return out

    if export_format == 'image':
        image_url = item_data.get('image_url') or (item_data.get('content') or {}).get('image_url')
        if not image_url or not str(image_url).startswith('data:image/svg+xml'):
            raise ValueError('No SVG image available for this item')
        svg_text = unquote(str(image_url).split(',', 1)[1])
        out = exports_dir / f'{base_name}.svg'
        out.write_text(svg_text, encoding='utf-8')
        return out

    raise ValueError('Unsupported export format')


def export_library_batch(items, export_format):
    exported = []
    for item in items:
        item_type = item.get('type')
        item_id = item.get('id')
        if item_type == 'exercise':
            model = get_or_404(Exercise, int(item_id), 'Exercise')
        elif item_type == 'exam':
            model = get_or_404(Exam, int(item_id), 'Exam')
        elif item_type == 'document':
            model = get_or_404(Document, int(item_id), 'Document')
        else:
            continue
        item_data = build_library_item(item_type, model)
        exported.append((item_type, item_data, export_item_file(item_type, item_data, export_format)))
    return exported


def build_moodle_xml(items, include_answers=True):
    rows = []
    rows.append('<?xml version="1.0" encoding="UTF-8"?>')
    rows.append('<quiz>')
    rows.append('<question type="category"><category><text>$course$/FrenchExerciseApp</text></category></question>')
    for item_type, item in items:
        if item_type != 'exercise':
            continue
        title = item.get('display_title', 'Exercise')
        content = item.get('content') or {}
        activity_type = content.get('activity_type', 'fill_blank')
        for idx, entry in enumerate(content.get('items', []), start=1):
            question_name = f'{title} #{idx}'
            if activity_type == 'matching':
                q_text = f"Relaciona: {entry.get('left', '')}"
                answer = entry.get('right', '')
            elif activity_type == 'dialogue':
                q_text = entry.get('line_with_blank', '')
                answer = entry.get('correct_answer', '')
            elif activity_type == 'color_match':
                q_text = f"Color de '{entry.get('word', '')}'"
                answer = entry.get('color_label', '')
            elif activity_type == 'image_choice':
                q_text = entry.get('question', '')
                answer = entry.get('correct_answer', '')
            elif activity_type == 'label_image':
                q_text = f"Describe: {entry.get('label', '')}"
                answer = entry.get('definition', '')
            elif activity_type == 'scene_story':
                q_text = entry.get('sentence', '')
                answer = str(entry.get('correct_order', ''))
            else:
                q_text = entry.get('question', '')
                answer = entry.get('correct_answer', '')
            rows.append('<question type="shortanswer">')
            rows.append(f'<name><text>{question_name}</text></name>')
            rows.append(f'<questiontext format="html"><text><![CDATA[{q_text}]]></text></questiontext>')
            if include_answers:
                rows.append('<answer fraction="100">')
                rows.append(f'<text>{answer}</text>')
                rows.append('</answer>')
            rows.append('</question>')
    rows.append('</quiz>')
    return '\n'.join(rows)


def build_h5p_payload(items, include_answers=True):
    activities = []
    for item_type, item in items:
        if item_type != 'exercise':
            continue
        content = item.get('content') or {}
        activities.append({
            'title': item.get('display_title'),
            'activity_type': content.get('activity_type', 'fill_blank'),
            'items': content.get('items', []),
            'include_answers': include_answers
        })
    return {'tool': 'FrenchExerciseApp', 'version': 1, 'activities': activities}


def register_download_file(path_obj):
    token = base64.urlsafe_b64encode(os.urandom(18)).decode('ascii').rstrip('=')
    EXPORT_TOKEN_MAP[token] = str(path_obj)
    return token


def restore_from_backup_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    data = payload.get('data', {})
    exercises = data.get('exercises', [])
    exams = data.get('exams', [])
    documents = data.get('documents', [])

    for e in exercises:
        content = e.get('content') or {'question': 'Sin contenido'}
        exists = Exercise.query.filter_by(
            title=e.get('title', 'Untitled'),
            topic=e.get('topic', 'general')
        ).first()
        if exists:
            continue
        item = Exercise(
            title=e.get('title', 'Untitled'),
            topic=e.get('topic', 'general'),
            level=e.get('level', 'A1'),
            exercise_type=e.get('exercise_type', 'fill_blank'),
            content=content,
            ai_model=e.get('ai_model', 'llama3')
        )
        db.session.add(item)

    for e in exams:
        exists = Exam.query.filter_by(title=e.get('title', 'Untitled exam')).first()
        if exists:
            continue
        item = Exam(
            title=e.get('title', 'Untitled exam'),
            description=e.get('description'),
            exercises=e.get('exercises') or [],
            total_score=e.get('total_score', 100)
        )
        db.session.add(item)

    for d in documents:
        exists = Document.query.filter_by(filename=d.get('filename', 'document')).first()
        if exists:
            continue
        item = Document(
            filename=d.get('filename', 'document'),
            analysis=d.get('analysis')
        )
        db.session.add(item)

    db.session.commit()
    return {
        'exercises': len(exercises),
        'exams': len(exams),
        'documents': len(documents)
    }


def auto_restore_if_empty():
    if not AUTO_RESTORE_FROM_BACKUP:
        return
    if Exercise.query.count() > 0:
        return
    _, backup_dir = get_backup_paths()
    latest_json = sorted(backup_dir.glob('backup_*.json'))
    if not latest_json:
        return
    stats = restore_from_backup_json(latest_json[-1])
    log_event(logging.INFO, 'backup.auto_restore.completed', source=str(latest_json[-1]), restored=stats)


def run_scheduled_backup():
    with app.app_context():
        result = export_backup()
        log_event(logging.INFO, 'backup.daily.created', sqlite_backup=result['sqlite_backup'])


def start_backup_scheduler():
    global scheduler
    if BackgroundScheduler is None:
        log_event(logging.WARNING, 'backup.scheduler.disabled', reason='apscheduler_not_installed')
        return
    if scheduler is not None:
        return
    scheduler = BackgroundScheduler()
    scheduler.add_job(run_scheduled_backup, 'cron', hour=BACKUP_HOUR, minute=BACKUP_MINUTE, id='daily_backup')
    scheduler.start()
    log_event(logging.INFO, 'backup.scheduler.started', hour=BACKUP_HOUR, minute=BACKUP_MINUTE)


def parse_ai_json_content(raw_text):
    if not raw_text:
        return {'question': 'Sin contenido generado'}

    text = raw_text.strip()

    # Remove fenced code block markers if present.
    if text.startswith('```'):
        lines = text.splitlines()
        if lines and lines[0].startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        text = '\n'.join(lines).strip()

    # Try parsing full text as JSON first.
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None

    # Try extracting first balanced JSON object/array if content has extra prose.
    if parsed is None:
        def extract_balanced_json(source):
            in_string = False
            escape = False
            stack = []
            start = None
            for i, ch in enumerate(source):
                if escape:
                    escape = False
                    continue
                if ch == '\\':
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch in '{[':
                    if start is None:
                        start = i
                    stack.append(ch)
                elif ch in '}]' and stack:
                    last = stack[-1]
                    if (last == '{' and ch == '}') or (last == '[' and ch == ']'):
                        stack.pop()
                        if not stack and start is not None:
                            return source[start:i + 1]
            return None

        snippet = extract_balanced_json(text)
        if snippet:
            try:
                parsed = json.loads(snippet)
            except Exception:
                parsed = None

    if isinstance(parsed, list):
        return {'items': parsed, 'question': f'{len(parsed)} ejercicios generados'}
    if isinstance(parsed, dict):
        return parsed
    return {'question': raw_text}


def topic_activity_hint(topic):
    t = (topic or '').lower()
    if any(key in t for key in ['couleur', 'color', 'colores']):
        return 'color_match'
    if any(key in t for key in ['téléphone', 'telefono', 'diálogo', 'dialogo']):
        return 'dialogue'
    if any(key in t for key in ['vêtement', 'ropa', 'objet', 'objets']):
        return 'image_choice'
    if any(key in t for key in ['histoire', 'story', 'cuento', 'rutina']):
        return 'scene_story'
    return None


def choose_activity_type(topic, requested_type):
    if requested_type in MAGIC_ACTIVITY_TYPES:
        return requested_type
    if requested_type == 'magic_mix':
        return next_magic_activity_type()
    hinted = topic_activity_hint(topic)
    return hinted or next_magic_activity_type()


def build_generation_prompt(topic, level, activity_type):
    template = PEDAGOGICAL_TEMPLATE_ENGINE.get(activity_type, PEDAGOGICAL_TEMPLATE_ENGINE['fill_blank'])
    required_fields = ', '.join(template['required_fields'])
    return f'''Eres diseñador pedagógico de francés para primaria.
Tema: "{topic}"
Nivel CEFR: {level}
Plantilla didáctica OBLIGATORIA: {template["name"]}
Objetivo docente: {template["teacher_goal"]}

Devuelve SOLO JSON válido (sin markdown, sin texto extra) con este esquema exacto:
{{
  "title": "string",
  "activity_type": "{activity_type}",
  "items": [{{ ... }}]
}}

Reglas estrictas:
- Debes generar entre 6 y 10 items.
- Cada item de tipo "{activity_type}" debe contener estos campos: {required_fields}
- Lenguaje apto para primaria.
- Evita frases repetidas.
- No incluyas explicaciones fuera de JSON.
'''


def evaluate_exercise_quality(content, expected_type=None):
    if not isinstance(content, dict):
        return {
            'score': 0.0,
            'passed': False,
            'reasons': ['content_not_object'],
            'metrics': {}
        }

    activity_type = content.get('activity_type', 'fill_blank')
    items = content.get('items') if isinstance(content.get('items'), list) else []
    expected_template = PEDAGOGICAL_TEMPLATE_ENGINE.get(expected_type or activity_type, PEDAGOGICAL_TEMPLATE_ENGINE['fill_blank'])
    required_fields = expected_template['required_fields']

    count_score = min(len(items) / 8, 1.0)
    type_score = 1.0 if (expected_type is None or activity_type == expected_type) else 0.0

    structure_hits = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        if all(field in item for field in required_fields):
            structure_hits += 1
    structure_score = (structure_hits / len(items)) if items else 0.0

    text_fragments = []
    for item in items:
        if not isinstance(item, dict):
            continue
        for key in ('question', 'line_with_blank', 'left', 'right', 'word', 'label', 'sentence', 'definition'):
            if item.get(key):
                text_fragments.append(str(item.get(key)).strip().lower())
    unique_ratio = (len(set(text_fragments)) / len(text_fragments)) if text_fragments else 0.0
    repetition_penalty = 0.0 if unique_ratio >= 0.7 else 0.2

    score = max(0.0, min(1.0, (0.30 * count_score) + (0.30 * structure_score) + (0.25 * unique_ratio) + (0.15 * type_score) - repetition_penalty))
    reasons = []
    if len(items) < 6:
        reasons.append('too_few_items')
    if structure_score < 0.7:
        reasons.append('schema_low_coverage')
    if unique_ratio < 0.7:
        reasons.append('low_variety')
    if expected_type and activity_type != expected_type:
        reasons.append('unexpected_activity_type')

    return {
        'score': round(score, 3),
        'passed': score >= QUALITY_MIN_SCORE,
        'reasons': reasons,
        'metrics': {
            'item_count': len(items),
            'count_score': round(count_score, 3),
            'structure_score': round(structure_score, 3),
            'variety_ratio': round(unique_ratio, 3),
            'type_score': round(type_score, 3)
        }
    }


def generate_structured_exercise(topic, level, exercise_type, ai_mode, ai_model):
    selected_type = choose_activity_type(topic, exercise_type)
    prompt = build_generation_prompt(topic, level, selected_type)
    raw = generate_with_provider(prompt, ai_mode=ai_mode, ai_model=ai_model)
    if not raw:
        payload = fallback_creative_content(topic, selected_type)
        payload = ensure_activity_structure(payload, forced_type=selected_type)
        quality = evaluate_exercise_quality(payload, expected_type=selected_type)
        return payload, quality, {'source': 'fallback_no_response', 'selected_type': selected_type}

    payload = ensure_activity_structure(parse_ai_json_content(raw), forced_type=selected_type)
    quality = evaluate_exercise_quality(payload, expected_type=selected_type)
    retries = 0
    while not quality['passed'] and retries < 2:
        retry_prompt = f'''Corrige la salida anterior.
Devuelve SOLO JSON válido con activity_type "{selected_type}" y entre 6 y 10 items.
No incluyas texto adicional fuera del JSON.
Tema: {topic}. Nivel: {level}.'''
        retried = generate_with_provider(retry_prompt, ai_mode=ai_mode, ai_model=ai_model)
        if not retried:
            break
        payload = ensure_activity_structure(parse_ai_json_content(retried), forced_type=selected_type)
        quality = evaluate_exercise_quality(payload, expected_type=selected_type)
        retries += 1

    if not quality['passed']:
        payload = ensure_activity_structure(fallback_creative_content(topic, selected_type), forced_type=selected_type)
        quality = evaluate_exercise_quality(payload, expected_type=selected_type)
        return payload, quality, {'source': 'fallback_low_quality', 'selected_type': selected_type}
    return payload, quality, {'source': 'provider', 'selected_type': selected_type}


def ensure_activity_structure(content, forced_type=None):
    data = content if isinstance(content, dict) else {'question': str(content)}
    items = data.get('items')
    activity_type = forced_type or data.get('activity_type') or 'fill_blank'

    if not isinstance(items, list):
        base_question = data.get('question', 'Completa la actividad')
        base_answer = data.get('correct_answer', 'mot')
        items = [{
            'question': base_question,
            'correct_answer': base_answer,
            'options': data.get('options', [base_answer, 'option 2', 'option 3']),
            'hint': data.get('hint', ''),
            'emoji': data.get('emoji', '📝')
        }]

    if activity_type == 'matching':
        normalized = []
        for idx, it in enumerate(items[:10]):
            left = it.get('left') or it.get('question') or f'élément {idx + 1}'
            right = it.get('right') or it.get('correct_answer') or f'option {idx + 1}'
            normalized.append({'left': left, 'right': right, 'emoji': it.get('emoji', '🧩')})
        data['items'] = normalized
    elif activity_type == 'color_match':
        palette = ['#1d4ed8', '#ef4444', '#22c55e', '#f59e0b', '#a855f7', '#06b6d4', '#f97316', '#84cc16']
        normalized = []
        for idx, it in enumerate(items[:12]):
            word = it.get('word') or it.get('correct_answer') or it.get('question') or f'couleur {idx + 1}'
            normalized.append({
                'word': word,
                'color_label': it.get('color_label') or it.get('correct_answer') or f'couleur {idx + 1}',
                'color_hex': it.get('color_hex') or palette[idx % len(palette)]
            })
        data['items'] = normalized
    elif activity_type == 'dialogue':
        normalized = []
        for idx, it in enumerate(items[:10]):
            normalized.append({
                'speaker': it.get('speaker') or f'Personnage {idx + 1}',
                'line_with_blank': it.get('line_with_blank') or it.get('question') or '....',
                'options': it.get('options') if isinstance(it.get('options'), list) else [],
                'correct_answer': it.get('correct_answer', ''),
                'emoji': it.get('emoji', '💬')
            })
        data['items'] = normalized
    elif activity_type == 'image_choice':
        normalized = []
        for idx, it in enumerate(items[:8]):
            raw_options = it.get('options') if isinstance(it.get('options'), list) else []
            option_labels = []
            for opt in raw_options[:4]:
                if isinstance(opt, dict):
                    option_labels.append(opt.get('label') or opt.get('text') or '')
                else:
                    option_labels.append(str(opt))
            while len(option_labels) < 4:
                option_labels.append(f'option {len(option_labels) + 1}')
            question = it.get('question') or f'Quel dessin correspond ? ({idx + 1})'
            choices = []
            for label in option_labels[:4]:
                choices.append({
                    'label': label,
                    'image_url': ensure_image_url(label, 'exercise')
                })
            normalized.append({
                'question': question,
                'choices': choices,
                'correct_answer': it.get('correct_answer') or option_labels[0],
                'hint': it.get('hint', ''),
                'emoji': it.get('emoji', '🖼️')
            })
        data['items'] = normalized
    elif activity_type == 'label_image':
        normalized = []
        for idx, it in enumerate(items[:10]):
            label = it.get('label') or it.get('word') or it.get('correct_answer') or f'objet {idx + 1}'
            normalized.append({
                'label': label,
                'definition': it.get('definition') or it.get('question') or 'Décris cet objet en français.',
                'image_url': ensure_image_url(it.get('image_url') or label, 'exercise'),
                'emoji': it.get('emoji', '🏷️')
            })
        data['items'] = normalized
    elif activity_type == 'scene_story':
        normalized = []
        for idx, it in enumerate(items[:8]):
            sentence = it.get('sentence') or it.get('question') or f'Étape {idx + 1}'
            normalized.append({
                'sentence': sentence,
                'correct_order': int(it.get('correct_order', idx + 1)),
                'image_url': ensure_image_url(it.get('image_url') or sentence, 'exercise'),
                'emoji': it.get('emoji', '🎬')
            })
        data['items'] = normalized
    else:
        activity_type = 'fill_blank'
        normalized = []
        for idx, it in enumerate(items[:12]):
            normalized.append({
                'question': it.get('question') or f'Complète la phrase {idx + 1}',
                'correct_answer': it.get('correct_answer') or 'mot',
                'options': it.get('options') if isinstance(it.get('options'), list) else ['option 1', 'option 2', 'option 3'],
                'hint': it.get('hint', ''),
                'emoji': it.get('emoji', '📝')
            })
        data['items'] = normalized

    data['activity_type'] = activity_type
    if 'title' not in data:
        data['title'] = 'Fiche créative'
    if 'image_url' not in data:
        data['image_url'] = ensure_image_url(data.get('title') or 'french worksheet', 'exercise')
    return data


def next_magic_activity_type():
    global MAGIC_ACTIVITY_INDEX
    activity = MAGIC_ACTIVITY_TYPES[MAGIC_ACTIVITY_INDEX % len(MAGIC_ACTIVITY_TYPES)]
    MAGIC_ACTIVITY_INDEX += 1
    return activity


def is_low_quality_content(content):
    if not isinstance(content, dict):
        return True
    items = content.get('items')
    if not isinstance(items, list) or len(items) < 4:
        return True
    raw_question = (content.get('question') or '').lower()
    noisy_markers = ['here is', 'let me know', '```', 'json worksheet']
    if any(marker in raw_question for marker in noisy_markers):
        return True
    return False


def fallback_creative_content(topic, activity_type):
    t = (topic or 'français').lower()
    title = f'Atelier créatif: {topic}'
    if activity_type == 'matching':
        pairs = [
            ('bonjour', 'hello'),
            ('au revoir', 'goodbye'),
            ('merci', 'thank you'),
            ('s’il vous plaît', 'please'),
            ('comment ça va ?', 'how are you?'),
            ('très bien', 'very well')
        ]
        if 'vêtement' in t:
            pairs = [('une robe', 'dress'), ('un pantalon', 'trousers'), ('un pull', 'sweater'), ('une jupe', 'skirt'), ('des chaussures', 'shoes'), ('une chemise', 'shirt')]
        return {
            'title': title,
            'activity_type': 'matching',
            'items': [{'left': l, 'right': r, 'emoji': '🧩'} for l, r in pairs]
        }

    if activity_type == 'color_match':
        colors = [
            ('rouge', 'rouge', '#ef4444'),
            ('bleu', 'bleu', '#2563eb'),
            ('vert', 'vert', '#16a34a'),
            ('jaune', 'jaune', '#facc15'),
            ('orange', 'orange', '#f97316'),
            ('violet', 'violet', '#9333ea'),
            ('rose', 'rose', '#ec4899'),
            ('noir', 'noir', '#111827')
        ]
        return {
            'title': title,
            'activity_type': 'color_match',
            'items': [{'word': w, 'color_label': label, 'color_hex': hex_color} for w, label, hex_color in colors]
        }

    if activity_type == 'dialogue':
        lines = [
            ('Lina', 'Bonjour, comment ______ ?', ['ça va', 'tu t’appelles', 'merci'], 'ça va', '☎️'),
            ('Samir', 'Je ______ Lina.', ['mange', 'm’appelle', 'écoute'], 'm’appelle', '☎️'),
            ('Lina', 'Tu habites où ______ France ?', ['au', 'en', 'de'], 'en', '🏠'),
            ('Samir', 'J’aime ______ français.', ['parler', 'manger', 'courir'], 'parler', '💬'),
            ('Lina', 'À bientôt et ______ !', ['merci', 'au revoir', 'bonjour'], 'au revoir', '👋'),
            ('Samir', 'Bonne ______ !', ['nuit', 'pomme', 'classe'], 'nuit', '🌙')
        ]
        return {
            'title': title,
            'activity_type': 'dialogue',
            'items': [
                {
                    'speaker': spk,
                    'line_with_blank': line,
                    'options': opts,
                    'correct_answer': ans,
                    'emoji': emo
                } for spk, line, opts, ans, emo in lines
            ]
        }

    if activity_type == 'image_choice':
        entries = [
            ('Quel vêtement porte-t-on en hiver ?', ['une écharpe', 'un maillot de bain', 'des sandales', 'un short'], 'une écharpe', 'il fait froid'),
            ('Quel objet utilise-t-on pour écrire ?', ['un stylo', 'une assiette', 'un ballon', 'une chaussure'], 'un stylo', 'objet scolaire'),
            ('Quel animal vit dans l’eau ?', ['le poisson', 'le cheval', 'le lapin', 'le chat'], 'le poisson', 'nage dans la mer'),
            ('Quel moyen de transport vole ?', ['l’avion', 'le bus', 'le vélo', 'le bateau'], 'l’avion', 'dans le ciel'),
            ('Quel aliment est sucré ?', ['le chocolat', 'le citron', 'le riz', 'la tomate'], 'le chocolat', 'dessert')
        ]
        return {
            'title': title,
            'activity_type': 'image_choice',
            'items': [
                {
                    'question': q,
                    'choices': [{'label': c, 'image_url': ensure_image_url(c, 'exercise')} for c in choices],
                    'correct_answer': answer,
                    'hint': hint,
                    'emoji': '🖼️'
                }
                for q, choices, answer, hint in entries
            ]
        }

    if activity_type == 'label_image':
        entries = [
            ('un pull', 'Vêtement chaud pour le haut du corps'),
            ('une jupe', 'Vêtement porté au niveau des jambes'),
            ('des baskets', 'Chaussures pour le sport'),
            ('un manteau', 'Protège du froid et de la pluie'),
            ('une chemise', 'Vêtement avec col et boutons'),
            ('un chapeau', 'Accessoire pour la tête')
        ]
        return {
            'title': title,
            'activity_type': 'label_image',
            'items': [
                {
                    'label': label,
                    'definition': definition,
                    'image_url': ensure_image_url(label, 'exercise'),
                    'emoji': '🏷️'
                }
                for label, definition in entries
            ]
        }

    if activity_type == 'scene_story':
        story = [
            'Je me réveille et je dis bonjour.',
            'Je prends mon petit-déjeuner.',
            'Je vais à l’école avec mon sac.',
            'Je joue avec mes amis dans la cour.',
            'Je rentre à la maison et je fais mes devoirs.',
            'Le soir, je lis un livre avant de dormir.'
        ]
        return {
            'title': title,
            'activity_type': 'scene_story',
            'items': [
                {
                    'sentence': sentence,
                    'correct_order': idx + 1,
                    'image_url': ensure_image_url(sentence, 'exercise'),
                    'emoji': '🎬'
                }
                for idx, sentence in enumerate(story)
            ]
        }

    # Default fill_blank fallback
    fill_items = [
        {'question': 'Je vois un ______ dans le ciel.', 'correct_answer': 'oiseau', 'options': ['oiseau', 'chat', 'table'], 'hint': 'animal qui vole', 'emoji': '🐦'},
        {'question': 'Le chat est de couleur ______.', 'correct_answer': 'noir', 'options': ['noir', 'bleu', 'orange'], 'hint': 'couleur sombre', 'emoji': '🐈'},
        {'question': 'Nous mangeons une ______ rouge.', 'correct_answer': 'pomme', 'options': ['pomme', 'jupe', 'voiture'], 'hint': 'fruit', 'emoji': '🍎'},
        {'question': 'À l’école, j’écris avec un ______.', 'correct_answer': 'stylo', 'options': ['stylo', 'banane', 'chaussure'], 'hint': 'objet de classe', 'emoji': '✏️'},
        {'question': 'Je porte un ______ quand il fait froid.', 'correct_answer': 'pull', 'options': ['pull', 'sandale', 'chapeau'], 'hint': 'vêtement chaud', 'emoji': '🧥'},
        {'question': 'Le soleil est ______ dans le dessin.', 'correct_answer': 'jaune', 'options': ['jaune', 'gris', 'noir'], 'hint': 'couleur du soleil', 'emoji': '☀️'}
    ]
    return {
        'title': title,
        'activity_type': 'fill_blank',
        'items': fill_items
    }


def test_provider(provider):
    probe_prompt = 'Responde solo: OK'
    def check_openai_compatible(api_key, base_url, model, key_name):
        if not api_key:
            return False, f'missing_api_key:{key_name}'
        try:
            response = requests.post(
                f'{base_url}/chat/completions',
                headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
                json={'model': model, 'messages': [{'role': 'user', 'content': probe_prompt}]},
                timeout=15
            )
            if response.status_code == 200:
                return True, 'reachable'
            if response.status_code == 429:
                return False, 'rate_limit'
            if response.status_code == 402:
                return False, 'insufficient_credits'
            return False, f'http_{response.status_code}'
        except Exception as e:
            return False, f'network_error:{str(e)}'

    if provider in ('llama3', 'mistral'):
        try:
            response = requests.post(
                f'{OLLAMA_URL}/api/generate',
                json={'model': provider, 'prompt': probe_prompt, 'stream': False},
                timeout=10
            )
            if response.status_code == 200:
                return True, 'reachable'
            return False, f'http_{response.status_code}'
        except Exception as e:
            return False, f'network_error:{str(e)}'

    if provider == 'perplexity':
        if not PERPLEXITY_KEY:
            return False, 'missing_api_key:PERPLEXITY_API_KEY'
        try:
            response = requests.post(
                'https://api.perplexity.ai/chat/completions',
                headers={'Authorization': f'Bearer {PERPLEXITY_KEY}', 'Content-Type': 'application/json'},
                json={'model': PERPLEXITY_MODEL, 'messages': [{'role': 'user', 'content': probe_prompt}]},
                timeout=15
            )
            if response.status_code == 200:
                return True, 'reachable'
            return False, f'http_{response.status_code}'
        except Exception as e:
            return False, f'network_error:{str(e)}'

    if provider == 'openai':
        return check_openai_compatible(OPENAI_KEY, 'https://api.openai.com/v1', OPENAI_MODEL, 'OPENAI_API_KEY')

    if provider == 'gemini':
        if not GEMINI_KEY:
            return False, 'missing_api_key:GEMINI_API_KEY'
        _, detail = call_gemini_generate(probe_prompt, timeout=15)
        if detail == 'reachable':
            return True, 'reachable'
        if detail.startswith('model_not_found:'):
            return False, 'gemini_model_not_found'
        if detail == 'auth_error':
            return False, 'gemini_auth_error'
        return False, detail

    if provider == 'deepseek':
        if not DEEPSEEK_KEY:
            return False, 'missing_api_key:DEEPSEEK_API_KEY'
        try:
            response = requests.post(
                'https://api.deepseek.com/chat/completions',
                headers={'Authorization': f'Bearer {DEEPSEEK_KEY}', 'Content-Type': 'application/json'},
                json={'model': DEEPSEEK_MODEL, 'messages': [{'role': 'user', 'content': probe_prompt}]},
                timeout=15
            )
            if response.status_code == 200:
                return True, 'reachable'
            if response.status_code == 402:
                return False, 'insufficient_credits'
            return False, f'http_{response.status_code}'
        except Exception as e:
            return False, f'network_error:{str(e)}'

    if provider == 'groq':
        return check_openai_compatible(GROQ_KEY, GROQ_BASE_URL, GROQ_MODEL, 'GROQ_API_KEY')

    if provider == 'glm':
        return check_openai_compatible(GLM_KEY, GLM_BASE_URL, GLM_MODEL, 'GLM_API_KEY')

    if provider == 'qwen':
        return check_openai_compatible(QWEN_KEY, QWEN_BASE_URL, QWEN_MODEL, 'QWEN_API_KEY')

    if provider == 'kimi':
        return check_openai_compatible(KIMI_KEY, KIMI_BASE_URL, KIMI_MODEL, 'KIMI_API_KEY')

    if provider == 'qwen_image':
        if not QWEN_KEY:
            return False, 'missing_api_key:QWEN_API_KEY'
        try:
            base = QWEN_BASE_URL.rstrip('/')
            response = requests.post(
                f'{base}/images/generations',
                headers={'Authorization': f'Bearer {QWEN_KEY}', 'Content-Type': 'application/json'},
                json={'model': QWEN_IMAGE_MODEL, 'prompt': 'test image', 'size': '512x512'},
                timeout=15
            )
            if response.status_code == 200:
                return True, 'reachable'
            return False, f'http_{response.status_code}'
        except Exception as e:
            return False, f'network_error:{str(e)}'

    if provider == 'wan':
        if not WAN_RUNTIME_URL:
            return False, 'missing_runtime:WAN_RUNTIME_URL'
        headers = {'Content-Type': 'application/json'}
        if WAN_API_KEY:
            headers['Authorization'] = f'Bearer {WAN_API_KEY}'
        try:
            response = requests.post(
                WAN_RUNTIME_URL,
                headers=headers,
                json={'prompt': probe_prompt, 'model': WAN_MODEL},
                timeout=12
            )
            if response.status_code in (200, 201, 202):
                return True, 'reachable'
            return False, f'http_{response.status_code}'
        except Exception as e:
            return False, f'network_error:{str(e)}'

    if provider == 'huggingface_ocr':
        if not HUGGINGFACE_KEY:
            return False, 'missing_api_key:HUGGINGFACE_API_KEY'
        return True, 'configured'

    if provider == 'kling':
        if not KLING_KEY:
            return False, 'missing_api_key:KLING_API_KEY'
        return True, 'configured'

    if provider == 'manus':
        if not MANUS_KEY:
            return False, 'missing_api_key:MANUS_API_KEY'
        return True, 'configured'

    if provider in {'sdxl', 'ltx_video', 'faster_whisper', 'piper_tts', 'paddleocr', 'instructor', 'guardrails', 'deepeval'}:
        module_map = {
            'sdxl': 'diffusers',
            'ltx_video': 'torch',
            'faster_whisper': 'faster_whisper',
            'piper_tts': 'piper',
            'paddleocr': 'paddleocr',
            'instructor': 'instructor',
            'guardrails': 'guardrails',
            'deepeval': 'deepeval'
        }
        installed = module_installed(module_map[provider])
        return (True, 'installed') if installed else (False, 'not_installed')

    if provider == 'promptfoo':
        installed = shutil.which('promptfoo') is not None
        return (True, 'installed') if installed else (False, 'not_installed')

    return False, 'unsupported_provider'


def ai_keys_health_snapshot():
    providers = {}
    missing_total = []
    configured_total = []
    for provider, required_vars in AI_KEY_REQUIREMENTS.items():
        missing_vars = [var for var in required_vars if not os.getenv(var, '').strip()]
        configured = len(missing_vars) == 0
        providers[provider] = {
            'status': 'green' if configured else 'red',
            'configured': configured,
            'missing_vars': missing_vars,
            'required_vars': required_vars
        }
        configured_total.extend([v for v in required_vars if v not in missing_vars])
        missing_total.extend(missing_vars)

    return {
        'summary': {
            'providers_ok': sum(1 for p in providers.values() if p['configured']),
            'providers_total': len(providers),
            'status': 'green' if len(missing_total) == 0 else 'red'
        },
        'providers': providers,
        'missing_vars': sorted(set(missing_total)),
        'configured_vars': sorted(set(configured_total))
    }

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'French Exercise App Backend is running'})


@app.route('/api/health/ai-keys', methods=['GET'])
def health_ai_keys():
    snapshot = ai_keys_health_snapshot()
    return jsonify({
        'tested_at': datetime.now(UTC).isoformat(),
        **snapshot
    })

@app.route('/api/exercises', methods=['GET'])
def get_exercises():
    exercises = Exercise.query.all()
    return jsonify([e.to_dict() for e in exercises])

@app.route('/api/exercises/generate', methods=['POST'])
def generate_exercise():
    data = request.get_json() or {}
    exercise_type = data.get('exercise_type', 'magic_mix')
    topic = data.get('topic', 'animals')
    level = data.get('level', 'A1')
    ai_mode = data.get('ai_mode', 'local')
    ai_model = data.get('ai_model', 'llama3')
    exercise_data, quality, generation_meta = generate_structured_exercise(
        topic=topic,
        level=level,
        exercise_type=exercise_type,
        ai_mode=ai_mode,
        ai_model=ai_model
    )
    if isinstance(exercise_data, dict):
        exercise_data['image_url'] = generate_illustration_data_uri(topic, 'exercise')
        exercise_data['quality'] = quality
    
    exercise = Exercise(
        title=f"{topic.capitalize()} - {exercise_type.replace('_', ' ').title()}",
        topic=topic,
        level=level,
        exercise_type=exercise_type,
        content=exercise_data,
        ai_model=ai_model
    )
    db.session.add(exercise)
    db.session.commit()
    audit_event('exercise.generate.single', {
        'topic': topic,
        'level': level,
        'requested_type': exercise_type,
        'activity_type': exercise_data.get('activity_type'),
        'quality_score': quality.get('score'),
        'quality_passed': quality.get('passed'),
        'source': generation_meta.get('source')
    })
    
    return jsonify(exercise.to_dict()), 201


@app.route('/api/exercises/generate-batch', methods=['POST'])
def generate_exercise_batch():
    if not feature_enabled('batch_generation'):
        return api_error(403, 'feature_disabled', 'batch_generation feature is disabled')

    data = request.get_json() or {}
    items = data.get('items') if isinstance(data.get('items'), list) else []
    if not items:
        return api_error(400, 'missing_items', 'items list is required')
    if len(items) > 100:
        return api_error(400, 'batch_too_large', 'Maximum batch size is 100 items')
    reject_low_quality = data.get('reject_low_quality', True)

    created = []
    errors = []
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            errors.append({'index': idx, 'error': 'item_must_be_object'})
            continue
        topic = (item.get('topic') or data.get('default_topic') or 'français').strip()
        level = item.get('level') or data.get('default_level') or 'A2'
        exercise_type = item.get('exercise_type') or data.get('default_exercise_type') or 'magic_mix'
        ai_mode = item.get('ai_mode') or data.get('default_ai_mode') or 'local'
        ai_model = item.get('ai_model') or data.get('default_ai_model') or 'llama3'
        payload, quality, generation_meta = generate_structured_exercise(
            topic=topic,
            level=level,
            exercise_type=exercise_type,
            ai_mode=ai_mode,
            ai_model=ai_model
        )
        batch_pass = quality.get('score', 0) >= QUALITY_BATCH_MIN_SCORE
        if reject_low_quality and not batch_pass:
            errors.append({
                'index': idx,
                'topic': topic,
                'error': 'quality_rejected',
                'quality': quality,
                'source': generation_meta.get('source')
            })
            continue
        payload['image_url'] = generate_illustration_data_uri(topic, 'exercise')
        payload['quality'] = quality
        exercise = Exercise(
            title=f"{topic.capitalize()} - {exercise_type.replace('_', ' ').title()}",
            topic=topic,
            level=level,
            exercise_type=exercise_type,
            content=payload,
            ai_model=ai_model if ai_mode != 'local' else ai_model
        )
        db.session.add(exercise)
        db.session.flush()
        created.append(exercise.to_dict())

    db.session.commit()
    audit_event('exercise.generate.batch', {'requested': len(items), 'created': len(created), 'errors': len(errors)})
    return jsonify({
        'ok': True,
        'requested': len(items),
        'created_count': len(created),
        'error_count': len(errors),
        'quality_threshold': QUALITY_BATCH_MIN_SCORE,
        'created': created,
        'errors': errors
    }), 201

@app.route('/api/exercises/<int:id>', methods=['GET'])
def get_exercise(id):
    exercise = get_or_404(Exercise, id, 'Exercise')
    return jsonify(exercise.to_dict())

@app.route('/api/exercises/<int:id>', methods=['DELETE'])
def delete_exercise(id):
    exercise = get_or_404(Exercise, id, 'Exercise')
    db.session.delete(exercise)
    db.session.commit()
    return jsonify({'message': 'Exercise deleted'})

@app.route('/api/exams', methods=['GET'])
def get_exams():
    exams = Exam.query.all()
    return jsonify([e.to_dict() for e in exams])

@app.route('/api/exams', methods=['POST'])
def create_exam():
    data = request.get_json()
    exam = Exam(
        title=data.get('title'),
        description=data.get('description'),
        exercises=data.get('exercises', []),
        total_score=data.get('total_score', 100)
    )
    db.session.add(exam)
    db.session.commit()
    return jsonify(exam.to_dict()), 201

@app.route('/api/exams/<int:id>', methods=['GET'])
def get_exam(id):
    exam = get_or_404(Exam, id, 'Exam')
    return jsonify(exam.to_dict())

@app.route('/api/exams/<int:id>', methods=['DELETE'])
def delete_exam(id):
    exam = get_or_404(Exam, id, 'Exam')
    db.session.delete(exam)
    db.session.commit()
    return jsonify({'message': 'Exam deleted'})


@app.route('/api/documents', methods=['GET'])
def get_documents():
    documents = Document.query.all()
    return jsonify([d.to_dict() for d in documents])

@app.route('/api/documents/upload', methods=['POST'])
def upload_document():
    if 'file' not in request.files:
        return api_error(400, 'missing_file', 'No file provided')
    
    file = request.files['file']
    if file.filename == '':
        return api_error(400, 'missing_file_name', 'No file selected')
    
    file_data = file.read()
    document = Document(filename=file.filename, file_data=file_data)
    db.session.add(document)
    db.session.commit()
    
    return jsonify(document.to_dict()), 201

@app.route('/api/documents/<int:id>/analyze', methods=['POST'])
def analyze_document(id):
    document = get_or_404(Document, id, 'Document')
    
    ai_mode = request.get_json().get('ai_mode', 'local') if request.is_json else 'local'
    ai_model = request.get_json().get('ai_model', 'llama3') if request.is_json else 'llama3'
    
    prompt = f'''Analyze this document and extract:
    1. Main topic
    2. Key information
    3. Vocabulary (if in French)
    4. Suggested exercises based on content
    
    Respond in JSON format.'''
    
    analysis = generate_with_provider(prompt, ai_mode=ai_mode, ai_model=ai_model)
    
    if analysis:
        try:
            analysis_data = json.loads(analysis) if '{' in analysis else {'analysis': analysis}
        except:
            analysis_data = {'analysis': analysis}
        
        document.analysis = analysis_data
        db.session.commit()
    
    return jsonify(document.to_dict())

@app.route('/api/ai/models', methods=['GET'])
def get_ai_models():
    models = {
        'local': ['llama3', 'mistral'],
        'cloud': [],
        'tools': []
    }
    if PERPLEXITY_KEY:
        models['cloud'].append('perplexity')
    if OPENAI_KEY:
        models['cloud'].append('openai')
    if GEMINI_KEY:
        models['cloud'].append('gemini')
    if DEEPSEEK_KEY:
        models['cloud'].append('deepseek')
    if GROQ_KEY:
        models['cloud'].append('groq')
    if GLM_KEY:
        models['cloud'].append('glm')
    if QWEN_KEY:
        models['cloud'].append('qwen')
    if KIMI_KEY:
        models['cloud'].append('kimi')
    if KLING_KEY:
        models['tools'].append('kling')
    if HUGGINGFACE_KEY:
        models['tools'].append('huggingface_ocr')
    if MANUS_KEY:
        models['tools'].append('manus')
    
    return jsonify(models)


@app.route('/api/ai/tools', methods=['GET'])
def get_ai_tools_catalog():
    local_tooling = {
        'sdxl': module_installed('diffusers'),
        'ltx_video': module_installed('torch'),
        'faster_whisper': module_installed('faster_whisper'),
        'piper_tts': module_installed('piper'),
        'paddleocr': module_installed('paddleocr'),
        'instructor': module_installed('instructor'),
        'guardrails': module_installed('guardrails'),
        'promptfoo': shutil.which('promptfoo') is not None,
        'deepeval': module_installed('deepeval')
    }
    items = [
        {
            'id': 'kling',
            'name': 'Kling AI',
            'category': 'video_generation',
            'url': 'https://klingai.com',
            'configured': bool(KLING_KEY),
            'status_detail': 'configured' if KLING_KEY else 'missing_api_key'
        },
        {
            'id': 'glm',
            'name': 'ChatGLM',
            'category': 'chat_vision_code',
            'url': 'https://chatglm.cn',
            'configured': bool(GLM_KEY),
            'status_detail': 'configured' if GLM_KEY else 'missing_api_key'
        },
        {
            'id': 'qwen',
            'name': 'Qwen',
            'category': 'long_document_analysis',
            'url': 'https://qwen.ai',
            'configured': bool(QWEN_KEY),
            'status_detail': 'configured' if QWEN_KEY else 'missing_api_key'
        },
        {
            'id': 'manus',
            'name': 'Manus AI',
            'category': 'autonomous_agent',
            'url': 'https://manus.im',
            'configured': bool(MANUS_KEY),
            'status_detail': 'configured' if MANUS_KEY else 'missing_api_key'
        },
        {
            'id': 'huggingface_ocr',
            'name': 'Hugging Face OCR',
            'category': 'ocr',
            'url': 'https://huggingface.co',
            'configured': bool(HUGGINGFACE_KEY),
            'status_detail': 'configured' if HUGGINGFACE_KEY else 'missing_api_key'
        },
        {
            'id': 'kimi',
            'name': 'Kimi (Moonshot)',
            'category': 'reasoning_assistant',
            'url': 'https://kimi.moonshot.cn',
            'configured': bool(KIMI_KEY),
            'status_detail': 'configured' if KIMI_KEY else 'missing_api_key'
        },
        {
            'id': 'qwen_image',
            'name': 'Qwen-Image',
            'category': 'image_generation',
            'url': 'https://huggingface.co/Qwen/Qwen-Image',
            'configured': bool(QWEN_KEY),
            'status_detail': 'configured' if QWEN_KEY else 'missing_api_key:QWEN_API_KEY'
        },
        {
            'id': 'wan',
            'name': 'Wan Video',
            'category': 'video_generation',
            'url': 'https://github.com/Wan-Video/Wan2.1',
            'configured': bool(WAN_RUNTIME_URL),
            'status_detail': 'configured' if WAN_RUNTIME_URL else 'pending_manual_runtime'
        },
        {
            'id': 'sdxl',
            'name': 'SDXL (Open Source)',
            'category': 'image_generation_local',
            'url': 'https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0',
            'configured': local_tooling['sdxl'],
            'status_detail': 'installed' if local_tooling['sdxl'] else 'not_installed'
        },
        {
            'id': 'ltx_video',
            'name': 'LTX-Video (Open Source)',
            'category': 'video_generation_local',
            'url': 'https://github.com/Lightricks/LTX-Video',
            'configured': local_tooling['ltx_video'],
            'status_detail': 'installed' if local_tooling['ltx_video'] else 'not_installed'
        },
        {
            'id': 'faster_whisper',
            'name': 'faster-whisper',
            'category': 'speech_to_text',
            'url': 'https://github.com/SYSTRAN/faster-whisper',
            'configured': local_tooling['faster_whisper'],
            'status_detail': 'installed' if local_tooling['faster_whisper'] else 'not_installed'
        },
        {
            'id': 'piper_tts',
            'name': 'Piper TTS',
            'category': 'text_to_speech',
            'url': 'https://github.com/rhasspy/piper',
            'configured': local_tooling['piper_tts'],
            'status_detail': 'installed' if local_tooling['piper_tts'] else 'not_installed'
        },
        {
            'id': 'paddleocr',
            'name': 'PaddleOCR',
            'category': 'ocr',
            'url': 'https://github.com/PaddlePaddle/PaddleOCR',
            'configured': local_tooling['paddleocr'],
            'status_detail': 'installed' if local_tooling['paddleocr'] else 'not_installed'
        },
        {
            'id': 'instructor',
            'name': 'Instructor JSON',
            'category': 'structured_output',
            'url': 'https://github.com/567-labs/instructor',
            'configured': local_tooling['instructor'],
            'status_detail': 'installed' if local_tooling['instructor'] else 'not_installed'
        },
        {
            'id': 'guardrails',
            'name': 'Guardrails',
            'category': 'output_validation',
            'url': 'https://github.com/guardrails-ai/guardrails',
            'configured': local_tooling['guardrails'],
            'status_detail': 'installed' if local_tooling['guardrails'] else 'not_installed'
        },
        {
            'id': 'promptfoo',
            'name': 'promptfoo',
            'category': 'quality_regression_tests',
            'url': 'https://github.com/promptfoo/promptfoo',
            'configured': local_tooling['promptfoo'],
            'status_detail': 'installed' if local_tooling['promptfoo'] else 'not_installed'
        },
        {
            'id': 'deepeval',
            'name': 'DeepEval',
            'category': 'quality_evaluation',
            'url': 'https://github.com/confident-ai/deepeval',
            'configured': local_tooling['deepeval'],
            'status_detail': 'installed' if local_tooling['deepeval'] else 'not_installed'
        }
    ]
    return jsonify({'items': items})


@app.route('/api/ai/tools/<tool_id>/test', methods=['POST'])
def test_ai_tool(tool_id):
    ok, detail = test_provider(tool_id)
    return jsonify({
        'tool_id': tool_id,
        'ok': ok,
        'status': 'green' if ok else 'red',
        'detail': detail
    })


@app.route('/api/ai/tools/<tool_id>/sample', methods=['POST'])
def sample_ai_tool(tool_id):
    data = request.get_json() or {}
    topic = (data.get('topic') or 'les vêtements').strip()
    level = (data.get('level') or 'A2').strip()
    if tool_id == 'kling':
        if not KLING_KEY:
            return jsonify({
                'tool_id': tool_id,
                'status': 'pending_manual',
                'detail': 'missing_api_key',
                'configure_url': 'https://klingai.com'
            }), 202
        return jsonify({
            'tool_id': tool_id,
            'status': 'queued',
            'provider': 'kling',
            'job_id': f'job_{int(time.time())}',
            'sample_prompt': f'Video educativo francés primaria, tema {topic}, nivel {level}, 30 segundos.'
        }), 202
    if tool_id in {'glm', 'qwen', 'kimi'}:
        provider_map = {'glm': 'glm', 'qwen': 'qwen', 'kimi': 'kimi'}
        prompt = f"Genera actividad creativa en francés para primaria. Tema: {topic}. Nivel: {level}. Salida JSON."
        text = generate_with_provider(prompt, ai_mode='cloud', ai_model=provider_map[tool_id])
        return jsonify({'tool_id': tool_id, 'sample': text or 'No response'}), 201
    if tool_id == 'qwen_image':
        image_url, resolved_provider = generate_image_asset(
            f"Ficha escolar de francés sobre {topic}, nivel {level}, estilo limpio para imprimir",
            provider='qwen_image'
        )
        return jsonify({
            'tool_id': tool_id,
            'provider': resolved_provider,
            'sample': 'Imagen de muestra generada',
            'image_url': image_url
        }), 201
    if tool_id == 'wan':
        if not WAN_RUNTIME_URL:
            return jsonify({
                'tool_id': tool_id,
                'status': 'pending_manual',
                'detail': 'missing_runtime:WAN_RUNTIME_URL',
                'configure_url': 'https://github.com/Wan-Video/Wan2.1'
            }), 202
        return jsonify({
            'tool_id': tool_id,
            'status': 'queued',
            'provider': 'wan',
            'job_id': f'wan_{int(time.time())}',
            'sample_prompt': f'Video educativo francés primaria, tema {topic}, nivel {level}, 20 segundos.'
        }), 202
    if tool_id == 'huggingface_ocr':
        return jsonify({
            'tool_id': tool_id,
            'sample': 'OCR sample pendiente de integración completa. Clave detectada correctamente.'
        }), 201
    if tool_id == 'manus':
        return jsonify({
            'tool_id': tool_id,
            'sample': 'Manus sample se gestiona externamente. Usa enlace oficial para tareas autónomas.'
        }), 201
    if tool_id in {'sdxl', 'ltx_video', 'faster_whisper', 'piper_tts', 'paddleocr', 'instructor', 'guardrails', 'promptfoo', 'deepeval'}:
        ok, detail = test_provider(tool_id)
        sample = {
            'sdxl': 'Prompt muestra: "Ficha escolar A4 sobre les vêtements, estilo ilustración infantil limpia".',
            'ltx_video': 'Storyboard: 3 escenas, 20s, vocabulario en clase con subtítulos FR/ES.',
            'faster_whisper': 'Transcripción: "Bonjour la classe..." -> texto para dictado.',
            'piper_tts': 'Audio local FR: instrucciones de actividad en voz docente.',
            'paddleocr': 'OCR de PDF: extraer enunciados para convertirlos en plantilla reutilizable.',
            'instructor': 'Validación de salida: forzar JSON por esquema de actividad.',
            'guardrails': 'Reglas: bloquear salida sin items mínimos o con texto fuera de JSON.',
            'promptfoo': 'Suite QA: comparar variedad/dificultad entre modelos.',
            'deepeval': 'Métrica lote: claridad, variedad, completitud y seguridad.'
        }
        return jsonify({
            'tool_id': tool_id,
            'ok': ok,
            'detail': detail,
            'sample': sample.get(tool_id, 'Sample disponible')
        }), 201
    return api_error(400, 'unsupported_tool', 'Tool not supported')


@app.route('/api/ai/tools/<tool_id>/diagnostic', methods=['GET'])
def diagnostic_ai_tool(tool_id):
    guidance = {
        'kling': [
            'Verificar KLING_API_KEY en backend/.env',
            'Comprobar plan activo para generación de video',
            'Probar primero sample con prompt corto (<= 200 caracteres)'
        ],
        'glm': [
            'Verificar GLM_API_KEY y GLM_BASE_URL',
            'Confirmar modelo GLM_MODEL habilitado',
            'Si falla con 401/403, regenerar API key y reiniciar backend'
        ],
        'qwen': [
            'Verificar QWEN_API_KEY y QWEN_BASE_URL',
            'Confirmar QWEN_MODEL disponible en tu cuenta',
            'Revisar cuota y límites de requests'
        ],
        'kimi': [
            'Verificar KIMI_API_KEY y KIMI_BASE_URL',
            'Comprobar KIMI_MODEL en tu tenant',
            'Probar test desde Dashboard IA Studio'
        ],
        'qwen_image': [
            'Verificar QWEN_API_KEY y endpoint compatible para imágenes',
            'Configurar QWEN_IMAGE_MODEL en .env',
            'Si falla, mantener fallback SVG para no interrumpir UX'
        ],
        'wan': [
            'Configurar WAN_RUNTIME_URL (servicio local/remoto de Wan)',
            'Opcional: WAN_API_KEY si el runtime exige autenticación',
            'Probar /api/media/video con provider=wan y prompt corto'
        ],
        'huggingface_ocr': [
            'Verificar HUGGINGFACE_API_KEY con permisos de Inference API',
            'Seleccionar modelo OCR soportado',
            'Probar con imagen pequeña antes de lotes grandes'
        ],
        'manus': [
            'Verificar MANUS_API_KEY',
            'Configurar scope de acciones/automatizaciones',
            'Usar prompts con objetivos y límites claros'
        ],
        'sdxl': [
            'Instalar `diffusers`, `transformers` y `torch` en backend/venv',
            'Configurar ruta/caché de modelo SDXL local o endpoint de inferencia',
            'Probar generación de 1 imagen antes de lotes grandes'
        ],
        'ltx_video': [
            'Instalar dependencias GPU/torch recomendadas por LTX-Video',
            'Configurar límite de duración (15-30s) para uso escolar',
            'Validar cola de jobs antes de habilitar en producción'
        ],
        'faster_whisper': [
            'Instalar `faster-whisper`',
            'Descargar modelo base/small para latencia aceptable',
            'Probar transcripción con audio corto (<30s)'
        ],
        'piper_tts': [
            'Instalar Piper y voz FR/ES',
            'Configurar ruta binario/voice model en .env',
            'Validar que audio se reproduce en navegador'
        ],
        'paddleocr': [
            'Instalar `paddleocr` y runtime compatible',
            'Probar OCR con PDF escaneado real',
            'Mapear resultado OCR a plantilla pedagógica'
        ],
        'instructor': [
            'Instalar `instructor`',
            'Definir esquema único de salida por activity_type',
            'Forzar validación antes de guardar en BD'
        ],
        'guardrails': [
            'Instalar `guardrails-ai`',
            'Definir reglas mínimas de calidad y formato',
            'Bloquear persistencia cuando no cumpla reglas'
        ],
        'promptfoo': [
            'Instalar CLI `promptfoo` (npm i -g promptfoo o npx)',
            'Crear suite con 20 prompts docentes reales',
            'Activar regresión en CI antes de desplegar'
        ],
        'deepeval': [
            'Instalar `deepeval`',
            'Definir métricas de claridad/variedad/ajuste nivel',
            'Ejecutar evaluación nocturna por lote'
        ]
    }
    checklist = guidance.get(tool_id)
    if not checklist:
        return api_error(404, 'tool_not_found', 'Tool diagnostic not found')
    return jsonify({'tool_id': tool_id, 'checklist': checklist})


@app.route('/api/ai/repair/gemini-deepseek', methods=['POST'])
def repair_gemini_deepseek():
    gemini = detect_best_gemini_models(timeout=8)
    fallback = recommend_fallback_provider()
    checklist = [
        f"Configurar GEMINI_MODELS={gemini['recommended_env_value']}",
        'Reiniciar backend para aplicar GEMINI_MODELS',
        f"Usar proveedor fallback recomendado: {fallback['provider']}",
        'Ejecutar /api/ai/test?force=1 para validar estado actualizado'
    ]
    return jsonify({
        'gemini': gemini,
        'deepseek_fallback': fallback,
        'checklist': checklist
    })


def generate_image_asset(prompt_text, provider='openai'):
    # Fallback always available so teacher never gets an empty result.
    fallback = generate_illustration_data_uri(prompt_text or 'french worksheet', 'exercise')
    if provider == 'qwen_image':
        if not QWEN_KEY:
            return fallback, 'qwen_image_fallback_missing_key'
        try:
            base = QWEN_BASE_URL.rstrip('/')
            headers = {
                'Authorization': f'Bearer {QWEN_KEY}',
                'Content-Type': 'application/json'
            }
            payload = {
                'model': QWEN_IMAGE_MODEL,
                'prompt': prompt_text,
                'size': '1024x1024'
            }
            response = requests.post(f'{base}/images/generations', headers=headers, json=payload, timeout=60)
            if response.status_code != 200:
                return fallback, f'qwen_image_http_{response.status_code}'
            data = response.json()
            item = (data.get('data') or [{}])[0]
            if item.get('b64_json'):
                return f"data:image/png;base64,{item['b64_json']}", 'qwen_image'
            if item.get('url'):
                return item['url'], 'qwen_image'
            return fallback, 'qwen_image_empty_payload'
        except Exception as e:
            log_event(logging.WARNING, 'provider.qwen_image.error', detail=str(e))
            return fallback, 'qwen_image_fallback_error'

    if provider != 'openai' or not OPENAI_KEY:
        return fallback, 'fallback_svg'
    try:
        headers = {
            'Authorization': f'Bearer {OPENAI_KEY}',
            'Content-Type': 'application/json'
        }
        payload = {
            'model': os.getenv('OPENAI_IMAGE_MODEL', 'gpt-image-1'),
            'prompt': prompt_text,
            'size': '1024x1024'
        }
        response = requests.post('https://api.openai.com/v1/images/generations', headers=headers, json=payload, timeout=60)
        if response.status_code != 200:
            return fallback, f'http_{response.status_code}'
        data = response.json()
        item = (data.get('data') or [{}])[0]
        if item.get('b64_json'):
            return f"data:image/png;base64,{item['b64_json']}", 'openai'
        if item.get('url'):
            return item['url'], 'openai'
    except Exception as e:
        log_event(logging.WARNING, 'provider.image.error', detail=str(e))
    return fallback, 'fallback_svg'


@app.route('/api/media/image', methods=['POST'])
def generate_media_image():
    data = request.get_json() or {}
    prompt = (data.get('prompt') or '').strip()
    if not prompt:
        return api_error(400, 'missing_prompt', 'prompt is required')
    provider = data.get('provider', 'openai')
    image_url, resolved_provider = generate_image_asset(prompt, provider=provider)
    return jsonify({
        'prompt': prompt,
        'provider': resolved_provider,
        'image_url': image_url
    }), 201


@app.route('/api/media/video', methods=['POST'])
def generate_media_video():
    data = request.get_json() or {}
    prompt = (data.get('prompt') or '').strip()
    if not prompt:
        return api_error(400, 'missing_prompt', 'prompt is required')
    provider = data.get('provider', 'kling')
    if provider == 'wan':
        if not WAN_RUNTIME_URL:
            return jsonify({
                'status': 'pending_manual',
                'provider': 'wan',
                'detail': 'missing_runtime:WAN_RUNTIME_URL',
                'configure_url': 'https://github.com/Wan-Video/Wan2.1'
            }), 202
        headers = {'Content-Type': 'application/json'}
        if WAN_API_KEY:
            headers['Authorization'] = f'Bearer {WAN_API_KEY}'
        payload = {'prompt': prompt, 'model': WAN_MODEL}
        try:
            response = requests.post(WAN_RUNTIME_URL, headers=headers, json=payload, timeout=20)
            if response.status_code in (200, 201, 202):
                data = response.json() if response.content else {}
                return jsonify({
                    'status': data.get('status', 'queued'),
                    'provider': 'wan',
                    'job_id': data.get('job_id') or data.get('id') or f"wan_{int(time.time())}",
                    'detail': data.get('detail', 'wan_request_accepted')
                }), 202
            return jsonify({
                'status': 'pending_manual',
                'provider': 'wan',
                'detail': f'wan_http_{response.status_code}',
                'configure_url': 'https://github.com/Wan-Video/Wan2.1'
            }), 202
        except Exception as e:
            log_event(logging.WARNING, 'provider.wan.error', detail=str(e))
            return jsonify({
                'status': 'pending_manual',
                'provider': 'wan',
                'detail': f'wan_unreachable:{str(e)}',
                'configure_url': 'https://github.com/Wan-Video/Wan2.1'
            }), 202
    if provider == 'kling' and not KLING_KEY:
        return jsonify({
            'status': 'pending_manual',
            'provider': 'kling',
            'detail': 'missing_api_key',
            'configure_url': 'https://klingai.com'
        }), 202
    # Placeholder async contract for future real integration.
    return jsonify({
        'status': 'queued',
        'provider': provider,
        'job_id': f'job_{int(time.time())}',
        'detail': 'Video generation request accepted'
    }), 202


def generate_audio_asset(text, voice='french_teacher'):
    # Reliable fallback audio (wav tone) so UI always has a playable asset.
    sample_rate = 22050
    duration_seconds = 1.6
    frequency = 523.25 if voice == 'french_teacher' else 440.0
    volume = 0.25
    n_samples = int(sample_rate * duration_seconds)

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        for i in range(n_samples):
            t = i / sample_rate
            value = int(32767 * volume * math.sin(2 * math.pi * frequency * t))
            wav_file.writeframes(struct.pack('<h', value))

    wav_b64 = base64.b64encode(wav_buffer.getvalue()).decode('utf-8')
    return f'data:audio/wav;base64,{wav_b64}', 'fallback_tone'


@app.route('/api/media/audio', methods=['POST'])
def generate_media_audio():
    data = request.get_json() or {}
    text = (data.get('text') or '').strip()
    if not text:
        return api_error(400, 'missing_text', 'text is required')
    voice = data.get('voice', 'french_teacher')

    audio_url, resolved_provider = generate_audio_asset(text, voice=voice)
    return jsonify({
        'text': text,
        'voice': voice,
        'provider': resolved_provider,
        'audio_url': audio_url
    }), 201


@app.route('/api/interactive/score', methods=['POST'])
def interactive_score():
    data = request.get_json() or {}
    answers = data.get('answers') if isinstance(data.get('answers'), list) else []
    correct = 0
    total = 0
    for item in answers:
        if not isinstance(item, dict):
            continue
        total += 1
        if str(item.get('selected')).strip().lower() == str(item.get('expected')).strip().lower():
            correct += 1
    percent = int((correct / total) * 100) if total else 0
    return jsonify({'correct': correct, 'total': total, 'percent': percent})


@app.route('/api/interactive/session', methods=['POST'])
def interactive_session():
    if not feature_enabled('interactive_sessions'):
        return api_error(403, 'feature_disabled', 'interactive_sessions feature is disabled')

    data = request.get_json() or {}
    exercise_id = data.get('exercise_id')
    student_id = data.get('student_id') or 'anonymous'
    if not exercise_id:
        return api_error(400, 'missing_exercise_id', 'exercise_id is required')
    exercise = get_or_404(Exercise, int(exercise_id), 'Exercise')
    content = exercise.content if isinstance(exercise.content, dict) else {}
    items = content.get('items') if isinstance(content.get('items'), list) else []
    session_id = str(uuid.uuid4())
    return jsonify({
        'session_id': session_id,
        'student_id': student_id,
        'exercise_id': exercise.id,
        'activity_type': content.get('activity_type', exercise.exercise_type),
        'total_items': len(items),
        'started_at': datetime.now(UTC).isoformat()
    }), 201


@app.route('/api/interactive/submit', methods=['POST'])
def interactive_submit():
    if not feature_enabled('interactive_sessions'):
        return api_error(403, 'feature_disabled', 'interactive_sessions feature is disabled')
    data = request.get_json() or {}
    answers = data.get('answers') if isinstance(data.get('answers'), list) else []
    session_id = data.get('session_id')
    if not session_id:
        return api_error(400, 'missing_session_id', 'session_id is required')

    correct = 0
    total = 0
    wrong_items = []
    for idx, item in enumerate(answers, start=1):
        if not isinstance(item, dict):
            continue
        total += 1
        selected = str(item.get('selected', '')).strip().lower()
        expected = str(item.get('expected', '')).strip().lower()
        ok = selected == expected and expected != ''
        if ok:
            correct += 1
        else:
            wrong_items.append({'index': idx, 'selected': item.get('selected'), 'expected': item.get('expected')})
    percent = int((correct / total) * 100) if total else 0
    return jsonify({
        'session_id': session_id,
        'correct': correct,
        'total': total,
        'percent': percent,
        'wrong_items': wrong_items
    })


@app.route('/api/ai/test', methods=['GET'])
def test_ai_providers():
    force_refresh = request.args.get('force') == '1'
    now = datetime.now(UTC)
    if not force_refresh and AI_TEST_CACHE['timestamp'] and AI_TEST_CACHE['payload']:
        if now - AI_TEST_CACHE['timestamp'] < timedelta(seconds=AI_TEST_CACHE_TTL_SECONDS):
            return jsonify(AI_TEST_CACHE['payload'])

    providers = [
        'llama3', 'mistral', 'perplexity', 'openai', 'gemini', 'deepseek',
        'groq', 'glm', 'qwen', 'kimi', 'qwen_image', 'wan', 'kling', 'huggingface_ocr', 'manus',
        'sdxl', 'ltx_video', 'faster_whisper', 'piper_tts', 'paddleocr', 'instructor', 'guardrails', 'promptfoo', 'deepeval'
    ]
    results = {}
    for provider in providers:
        ok, detail = test_provider(provider)
        results[provider] = {
            'ok': ok,
            'status': 'green' if ok else 'red',
            'detail': detail
        }
    payload = {
        'tested_at': datetime.now(UTC).isoformat(),
        'results': results
    }
    AI_TEST_CACHE['timestamp'] = now
    AI_TEST_CACHE['payload'] = payload
    audit_event('ai.test', {'providers': providers})
    return jsonify(payload)


@app.route('/api/ai/providers/<provider>/refine', methods=['POST'])
def refine_with_provider(provider):
    data = request.get_json() or {}
    text = data.get('text', '')
    if not text:
        return api_error(400, 'missing_text', 'text is required')
    if provider not in {'openai', 'gemini', 'perplexity', 'deepseek', 'llama3', 'mistral', 'groq', 'glm', 'qwen', 'kimi'}:
        return api_error(400, 'unsupported_provider', 'Unsupported provider')

    ai_mode = 'local' if provider in {'llama3', 'mistral'} else 'cloud'
    prompt = f"""Refina este material didáctico de francés para primaria.
Mantén formato JSON cuando exista y mejora claridad y creatividad.
Texto:
{text}
"""
    refined = generate_with_provider(prompt, ai_mode=ai_mode, ai_model=provider)
    if not refined:
        return api_error(502, 'provider_no_response', 'No response from provider')
    audit_event('ai.refine', {'provider': provider})
    return jsonify({'provider': provider, 'result': refined})


@app.route('/api/chat/messages', methods=['GET'])
def get_chat_messages():
    limit = max(1, min(int(request.args.get('limit', '100')), 500))
    rows = ChatMessage.query.order_by(ChatMessage.id.desc()).limit(limit).all()
    return jsonify([row.to_dict() for row in reversed(rows)])


def build_chat_prompt(task_type, message, context):
    task_type = task_type or 'chat'
    if task_type == 'interactive_gen':
        topic = context.get('topic', 'français')
        level = context.get('level', 'A2')
        return f"""Create an INTERACTIVE French classroom activity in JSON.
Topic: {topic}
Level: {level}
User request: {message}
Return valid JSON with fields:
title, activity_type, instructions, items[], scoring.
Use activity_type as one of: matching, color_match, dialogue, drag_drop, quiz_live."""
    if task_type == 'image_gen':
        topic = context.get('topic', 'français')
        return f"""Create a concise visual prompt in Spanish for an educational image.
Topic: {topic}
User request: {message}
Return JSON only: {{"title":"", "image_prompt":"", "style":"children_worksheet"}}"""
    if task_type == 'video_gen':
        topic = context.get('topic', 'français')
        return f"""Create a short storyboard for classroom video generation.
Topic: {topic}
User request: {message}
Return JSON only with fields: title, duration_seconds, scenes[]."""
    if task_type == 'exercise_gen':
        topic = context.get('topic', 'français')
        level = context.get('level', 'A2')
        return f"""Eres diseñador pedagógico experto en francés para primaria.
Tema base: {topic}
Nivel: {level}
Petición docente: {message}

Objetivo:
- Genera una actividad creativa (no repetitiva, evita caer siempre en fill_blank).
- Elige de forma inteligente activity_type entre:
  fill_blank, matching, color_match, dialogue, image_choice, label_image, scene_story
- Si la petición pide dinamismo/juego, prioriza formatos interactivos (dialogue, image_choice, scene_story, matching).

Devuelve SOLO JSON válido con forma:
{{
  "title": "string",
  "activity_type": "string",
  "items": [{{...}}]
}}

Reglas:
- 6 a 10 items
- lenguaje de primaria
- variedad alta y consignas claras
- nada de texto fuera del JSON
"""
    if task_type == 'exam_gen':
        topic = context.get('topic', 'français')
        return f"""Create a short French exam plan for primary school.
Topic: {topic}
User request: {message}
Return ONLY valid JSON with: title, description, exercises[], total_score.
No prose outside JSON."""
    if task_type == 'doc_analysis':
        return f"""Analyze educational content and suggest classroom activities.
User request: {message}
Return concise JSON with summary, key_points, suggested_activities."""
    return f"""You are a French teacher assistant for primary school.
Respond in Spanish when explaining and include French examples when relevant.
User message: {message}"""


def detect_game_intent(user_message):
    text = (user_message or '').lower()
    if any(token in text for token in ['ruleta', 'roulette', 'girar']):
        return 'wheel'
    if any(token in text for token in ['memoria', 'memory', 'parejas']):
        return 'memory_pairs'
    if any(token in text for token in ['quiz', 'trivial', 'concurso']):
        return 'quiz_live'
    if any(token in text for token in ['rol', 'roleplay', 'diálogo']):
        return 'roleplay'
    if any(token in text for token in ['juego', 'game', 'interactivo']):
        return 'game_mix'
    return None


def build_ui_game_preview(preview, game_intent):
    items = preview.get('items') if isinstance(preview.get('items'), list) else []
    labels = []
    for item in items[:10]:
        labels.append(
            item.get('question')
            or item.get('line_with_blank')
            or item.get('left')
            or item.get('word')
            or item.get('label')
            or item.get('sentence')
            or 'Reto'
        )
    if not labels:
        labels = ['Describe une couleur', 'Décris une tenue', 'Invente un dialogue', 'Trouve un synonyme']

    if game_intent == 'wheel':
        return {
            'type': 'wheel',
            'title': 'Ruleta de retos',
            'segments': labels[:8],
            'rules': [
                'Cada alumno gira una vez',
                'Debe resolver el reto en 30 segundos',
                'El grupo valida la respuesta'
            ]
        }
    if game_intent == 'memory_pairs':
        return {
            'type': 'memory_pairs',
            'title': 'Juego de memoria por parejas',
            'pairs_preview': labels[:6]
        }
    if game_intent == 'quiz_live':
        return {
            'type': 'quiz_live',
            'title': 'Quiz en vivo',
            'rounds': min(8, max(4, len(labels)))
        }
    if game_intent == 'roleplay':
        return {
            'type': 'roleplay',
            'title': 'Roleplay guiado',
            'roles': ['A', 'B'],
            'situations': labels[:5]
        }
    return {
        'type': 'game_mix',
        'title': 'Juego creativo mixto',
        'challenges': labels[:8]
    }


def enrich_chat_preview(preview, user_message, context):
    if not isinstance(preview, dict):
        return preview
    game_intent = detect_game_intent(user_message)
    if game_intent:
        preview['ui_game'] = build_ui_game_preview(preview, game_intent)
    preview['assistant_guidance'] = {
        'steps': [
            'Revisa la vista previa y valida nivel/tema',
            'Pulsa una mejora rápida si quieres ajustes',
            'Guarda como ejercicio o examen cuando esté listo'
        ],
        'teacher_tip': 'Pide cambios concretos: más visual, más oral, más juego por equipos.'
    }
    preview['improvement_suggestions'] = [
        'Hazlo más visual',
        'Añade dinámica por equipos',
        'Incluye autocorrección',
        'Crea versión alumno y profesor',
        'Añade audio de instrucciones'
    ]
    preview['deliverables'] = {
        'exercise': True,
        'exam': True,
        'image': True,
        'video': True,
        'audio': True,
        'pdf': True,
        'interactive': True
    }
    preview['requested_topic'] = context.get('topic', 'français') if isinstance(context, dict) else 'français'
    return preview


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json() or {}
    message = (data.get('message') or '').strip()
    if not message:
        return api_error(400, 'missing_message', 'message is required')

    task_type = data.get('task_type', 'chat')
    provider = data.get('provider', 'llama3')
    model = data.get('model', provider)
    ai_mode = 'local' if provider in {'llama3', 'mistral'} else 'cloud'
    context = data.get('context') if isinstance(data.get('context'), dict) else {}

    user_row = ChatMessage(
        role='user',
        content=message,
        task_type=task_type,
        provider=provider,
        model=model
    )
    db.session.add(user_row)

    prompt = build_chat_prompt(task_type, message, context)
    structured_preview = None
    answer = generate_with_provider(prompt, ai_mode=ai_mode, ai_model=model or provider)
    if not answer:
        answer = 'No fue posible obtener respuesta del modelo en este momento.'
    elif task_type in {'exercise_gen', 'interactive_gen'}:
        topic = context.get('topic', 'français')
        parsed = parse_ai_json_content(answer)
        model_activity = parsed.get('activity_type') if isinstance(parsed, dict) else None
        selected_type = model_activity if model_activity in MAGIC_ACTIVITY_TYPES else choose_activity_type(topic, 'magic_mix')
        normalized = ensure_activity_structure(parsed, forced_type=selected_type if model_activity in MAGIC_ACTIVITY_TYPES else None)
        if normalized.get('activity_type') == 'fill_blank':
            fallback_creative_type = next_magic_activity_type()
            if fallback_creative_type == 'fill_blank':
                fallback_creative_type = 'dialogue'
            normalized = ensure_activity_structure(fallback_creative_content(topic, fallback_creative_type), forced_type=fallback_creative_type)
            selected_type = fallback_creative_type
        quality = evaluate_exercise_quality(normalized, expected_type=None)
        if not quality.get('passed'):
            normalized = ensure_activity_structure(fallback_creative_content(topic, selected_type), forced_type=selected_type)
            quality = evaluate_exercise_quality(normalized, expected_type=None)
        normalized['quality'] = quality
        structured_preview = enrich_chat_preview(normalized, message, context)
        answer = json.dumps(normalized, ensure_ascii=False)
    elif task_type == 'exam_gen':
        parsed = parse_ai_json_content(answer)
        if not isinstance(parsed, dict):
            parsed = {}
        parsed.setdefault('title', f'Examen · {context.get("topic", "français")}')
        parsed.setdefault('description', 'Evaluación para primaria')
        parsed.setdefault('exercises', [])
        parsed.setdefault('total_score', 100)
        answer = json.dumps(parsed, ensure_ascii=False)
        structured_preview = enrich_chat_preview(parsed, message, context)

    assistant_row = ChatMessage(
        role='assistant',
        content=answer,
        task_type=task_type,
        provider=provider,
        model=model
    )
    db.session.add(assistant_row)
    db.session.commit()
    audit_event('chat.message', {'task_type': task_type, 'provider': provider, 'model': model})

    return jsonify({
        'message': assistant_row.to_dict(),
        'preview': structured_preview,
        'meta': {
            'task_type': task_type,
            'provider': provider,
            'model': model,
            'ai_mode': ai_mode
        }
    }), 201


@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    data = request.get_json() or {}
    message = (data.get('message') or '').strip()
    if not message:
        return api_error(400, 'missing_message', 'message is required')

    task_type = data.get('task_type', 'chat')
    provider = data.get('provider', 'llama3')
    model = data.get('model', provider)
    ai_mode = 'local' if provider in {'llama3', 'mistral'} else 'cloud'
    context = data.get('context') if isinstance(data.get('context'), dict) else {}

    user_row = ChatMessage(
        role='user',
        content=message,
        task_type=task_type,
        provider=provider,
        model=model
    )
    db.session.add(user_row)
    db.session.commit()

    prompt = build_chat_prompt(task_type, message, context)

    @stream_with_context
    def event_stream():
        chunks = []
        try:
            for token in generate_with_provider_stream(prompt, ai_mode=ai_mode, ai_model=model or provider):
                chunks.append(token)
                payload = json.dumps({'token': token}, ensure_ascii=False)
                yield f'event: token\ndata: {payload}\n\n'

            answer = ''.join(chunks).strip() or 'No fue posible obtener respuesta del modelo en este momento.'
            assistant_row = ChatMessage(
                role='assistant',
                content=answer,
                task_type=task_type,
                provider=provider,
                model=model
            )
            db.session.add(assistant_row)
            db.session.commit()
            audit_event('chat.message.stream', {'task_type': task_type, 'provider': provider, 'model': model})
            done_payload = json.dumps({'message': assistant_row.to_dict()}, ensure_ascii=False)
            yield f'event: done\ndata: {done_payload}\n\n'
        except Exception as e:
            db.session.rollback()
            err_payload = json.dumps({'error': str(e)}, ensure_ascii=False)
            yield f'event: error\ndata: {err_payload}\n\n'

    return Response(event_stream(), mimetype='text/event-stream')


@app.route('/api/chat/convert', methods=['POST'])
def convert_chat_message():
    data = request.get_json() or {}
    chat_message_id = data.get('chat_message_id')
    target = data.get('target', 'exercise')
    if not chat_message_id:
        return api_error(400, 'missing_chat_message_id', 'chat_message_id is required')
    if target not in {'exercise', 'exam'}:
        return api_error(400, 'unsupported_target', 'target must be exercise or exam')

    row = get_or_404(ChatMessage, int(chat_message_id), 'ChatMessage')
    parsed = parse_ai_json_content(row.content)
    normalized = ensure_activity_structure(parsed)

    if target == 'exercise':
        topic = (data.get('topic') or 'chat_topic').strip() or 'chat_topic'
        level = data.get('level', 'A2')
        exercise_type = normalized.get('activity_type', 'fill_blank')
        normalized['image_url'] = generate_illustration_data_uri(topic, 'exercise')
        exercise = Exercise(
            title=f'{topic.capitalize()} - Chat',
            topic=topic,
            level=level,
            exercise_type=exercise_type,
            content=normalized,
            ai_model=row.model or 'llama3'
        )
        db.session.add(exercise)
        db.session.commit()
        audit_event('chat.convert.exercise', {'chat_message_id': row.id, 'exercise_id': exercise.id})
        return jsonify({'target': 'exercise', 'item': exercise.to_dict()}), 201

    exam_payload = parsed if isinstance(parsed, dict) else {}
    exercises = exam_payload.get('exercises')
    if not isinstance(exercises, list) or len(exercises) == 0:
        exercises = [{'title': normalized.get('title', 'Actividad'), 'content': normalized}]
    exam = Exam(
        title=exam_payload.get('title') or 'Examen desde chat',
        description=exam_payload.get('description') or 'Generado desde conversación IA',
        exercises=exercises,
        total_score=int(exam_payload.get('total_score', 100))
    )
    db.session.add(exam)
    db.session.commit()
    audit_event('chat.convert.exam', {'chat_message_id': row.id, 'exam_id': exam.id})
    return jsonify({'target': 'exam', 'item': exam.to_dict()}), 201

# --- Compatibility wrappers for /api/v1/chat ---
@app.route('/api/v1/chat/conversations', methods=['GET'])
def v1_get_chat_conversations():
    # Alias for legacy clients expecting /api/v1/chat/conversations
    return get_chat_messages()


@app.route('/api/v1/chat', methods=['POST'])
def v1_chat():
    # Alias to main `/api/chat` handler
    return chat()


@app.route('/api/v1/chat/stream', methods=['POST'])
def v1_chat_stream():
    # Alias to streaming endpoint
    return chat_stream()


@app.route('/api/backups/export', methods=['POST'])
def export_backup_now():
    result = export_backup()
    return jsonify(result), 201


@app.route('/api/backups/restore-latest', methods=['POST'])
def restore_latest_backup():
    _, backup_dir = get_backup_paths()
    latest_json = sorted(backup_dir.glob('backup_*.json'))
    if not latest_json:
        return api_error(404, 'backup_not_found', 'No backup files found')
    stats = restore_from_backup_json(latest_json[-1])
    return jsonify({
        'restored_from': str(latest_json[-1]),
        'restored_counts': stats
    }), 201


@app.route('/api/library/items', methods=['GET'])
def library_items():
    items = []
    items.extend([build_library_item('exercise', e) for e in Exercise.query.order_by(Exercise.id.desc()).all()])
    items.extend([build_library_item('exam', e) for e in Exam.query.order_by(Exam.id.desc()).all()])
    items.extend([build_library_item('document', d) for d in Document.query.order_by(Document.id.desc()).all()])
    items.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    return jsonify(items)


@app.route('/api/library/search/semantic', methods=['GET'])
def library_semantic_search():
    if not feature_enabled('semantic_search'):
        return api_error(403, 'feature_disabled', 'semantic_search feature is disabled')

    query = (request.args.get('q') or '').strip().lower()
    limit = max(1, min(int(request.args.get('limit', '25')), 100))
    if not query:
        return api_error(400, 'missing_query', 'q query parameter is required')

    query_tokens = [tok for tok in query.replace(',', ' ').replace('.', ' ').split() if tok]
    if not query_tokens:
        return api_error(400, 'invalid_query', 'query must contain searchable tokens')

    items = []
    items.extend([build_library_item('exercise', e) for e in Exercise.query.order_by(Exercise.id.desc()).all()])
    items.extend([build_library_item('exam', e) for e in Exam.query.order_by(Exam.id.desc()).all()])
    items.extend([build_library_item('document', d) for d in Document.query.order_by(Document.id.desc()).all()])

    scored = []
    for item in items:
        hay = json.dumps(item, ensure_ascii=False).lower()
        token_hits = sum(1 for tok in query_tokens if tok in hay)
        if token_hits == 0:
            continue
        score = round(token_hits / max(1, len(query_tokens)), 3)
        item['_semantic_score'] = score
        scored.append(item)

    scored.sort(key=lambda row: (row.get('_semantic_score', 0), row.get('created_at', '')), reverse=True)
    return jsonify({
        'query': query,
        'count': len(scored[:limit]),
        'results': scored[:limit]
    })


def _run_repair_exercises():
    repaired = 0
    for ex in Exercise.query.all():
        content = ex.content if isinstance(ex.content, dict) else {}
        needs_repair = False
        if not isinstance(content.get('items'), list):
            q = content.get('question')
            if isinstance(q, str) and ('{' in q or '[' in q):
                needs_repair = True
        if needs_repair:
            parsed = parse_ai_json_content(content.get('question', ''))
            normalized = ensure_activity_structure(parsed)
            normalized['image_url'] = generate_illustration_data_uri(ex.topic, 'exercise')
            ex.content = normalized
            repaired += 1
    db.session.commit()
    return {'repaired': repaired}


@app.route('/api/library/repair-exercises', methods=['POST'])
def repair_exercises():
    # Legacy endpoint kept for backward compatibility. Prefer /api/exercises/repair-batch.
    response = jsonify(_run_repair_exercises())
    response.headers['X-Deprecated-Endpoint'] = '/api/library/repair-exercises'
    response.headers['X-Replacement-Endpoint'] = '/api/exercises/repair-batch'
    return response


@app.route('/api/library/import-francais6', methods=['POST'])
def import_francais6_materials():
    data = request.get_json() or {}
    import_root = Path(data.get('import_root') or FRANCAIS6_IMPORT_ROOT).expanduser()
    dry_run = bool(data.get('dry_run', False))
    max_files = max(1, min(int(data.get('max_files', 200)), 1000))

    if not import_root.exists() or not import_root.is_dir():
        return api_error(404, 'import_root_not_found', f'Import root not found: {import_root}')

    allowed_exts = {'.pdf', '.odt'}
    discovered = []
    for path_obj in import_root.rglob('*'):
        if len(discovered) >= max_files:
            break
        if not path_obj.is_file():
            continue
        if path_obj.name.startswith('._'):
            continue
        if path_obj.suffix.lower() not in allowed_exts:
            continue
        discovered.append(path_obj)

    existing_sources = set()
    for ex in Exercise.query.all():
        if not isinstance(ex.content, dict):
            continue
        src = (((ex.content or {}).get('import_metadata') or {}).get('source_path'))
        if isinstance(src, str) and src:
            existing_sources.add(src)

    created = []
    skipped = []

    for source in discovered:
        source_str = str(source)
        if source_str in existing_sources:
            skipped.append({'source_path': source_str, 'reason': 'already_imported'})
            continue

        topic, activity_type = infer_template_topic_and_type(source)
        template_title = f"Plantilla · {source.stem}"
        content = build_imported_template_content(source_str, topic, activity_type, template_title)

        if dry_run:
            created.append({
                'title': template_title,
                'topic': topic,
                'exercise_type': activity_type,
                'source_path': source_str
            })
            continue

        exercise = Exercise(
            title=template_title,
            topic=topic,
            level='A2',
            exercise_type=activity_type,
            content=content,
            ai_model='template_import'
        )
        db.session.add(exercise)
        created.append({
            'title': template_title,
            'topic': topic,
            'exercise_type': activity_type,
            'source_path': source_str
        })
        existing_sources.add(source_str)

    if not dry_run:
        db.session.commit()

    audit_event('library.import.francais6', {
        'import_root': str(import_root),
        'discovered': len(discovered),
        'created': len(created),
        'skipped': len(skipped),
        'dry_run': dry_run
    })
    return jsonify({
        'import_root': str(import_root),
        'dry_run': dry_run,
        'discovered_count': len(discovered),
        'created_count': len(created),
        'skipped_count': len(skipped),
        'created': created[:120],
        'skipped': skipped[:120]
    }), 201


@app.route('/api/library/items/<item_type>/<int:item_id>', methods=['PUT'])
def update_library_item(item_type, item_id):
    data = request.get_json() or {}
    if item_type == 'exercise':
        item = get_or_404(Exercise, item_id, 'Exercise')
        item.title = data.get('title', item.title)
        item.topic = data.get('topic', item.topic)
        if 'content' in data and isinstance(data['content'], dict):
            item.content = data['content']
    elif item_type == 'exam':
        item = get_or_404(Exam, item_id, 'Exam')
        item.title = data.get('title', item.title)
        item.description = data.get('description', item.description)
        if 'exercises' in data and isinstance(data['exercises'], list):
            item.exercises = data['exercises']
        item.total_score = int(data.get('total_score', item.total_score or 100))
    elif item_type == 'document':
        item = get_or_404(Document, item_id, 'Document')
        item.filename = data.get('filename', item.filename)
        if 'analysis' in data:
            item.analysis = data['analysis']
    else:
        return api_error(400, 'unsupported_item_type', 'Unsupported item type')

    db.session.commit()
    return jsonify(build_library_item(item_type, item))


@app.route('/api/library/export', methods=['POST'])
def export_library_item():
    data = request.get_json() or {}
    item_type = data.get('item_type')
    item_id = data.get('item_id')
    export_format = data.get('format', 'json')
    options = data.get('options') if isinstance(data.get('options'), dict) else {}
    if not item_type or not item_id:
        return api_error(400, 'missing_item_identifiers', 'item_type and item_id are required')

    if item_type == 'exercise':
        item = get_or_404(Exercise, int(item_id), 'Exercise')
    elif item_type == 'exam':
        item = get_or_404(Exam, int(item_id), 'Exam')
    elif item_type == 'document':
        item = get_or_404(Document, int(item_id), 'Document')
    else:
        return api_error(400, 'unsupported_item_type', 'Unsupported item type')

    item_data = build_library_item(item_type, item)
    try:
        out_file = export_item_file(item_type, item_data, export_format, options=options)
    except Exception as e:
        return api_error(400, 'export_error', str(e))

    token = register_download_file(out_file)
    audit_event('library.export', {'item_type': item_type, 'item_id': item_id, 'format': export_format})
    payload = {
        'path': str(out_file),
        'filename': out_file.name,
        'format': export_format,
        'download_token': token,
        'download_url': f'/api/library/export/download/{token}'
    }
    if export_format == 'google_workspace':
        payload['workspace_links'] = {
            'drive_upload': 'https://drive.google.com/drive/u/0/my-drive',
            'docs_new': 'https://docs.new',
            'classroom': 'https://classroom.google.com/'
        }
        payload['notes'] = [
            'Sube el archivo .gworkspace.html a Google Drive',
            'Abre con Google Docs para edición colaborativa',
            'Comparte el enlace en Google Classroom'
        ]
    return jsonify(payload), 201


@app.route('/api/google/workspace/health', methods=['GET'])
def google_workspace_health():
    ready, reason = google_workspace_ready()
    return jsonify({
        'ready': ready,
        'reason': reason,
        'service_account_file_configured': bool(GOOGLE_SERVICE_ACCOUNT_FILE),
        'service_account_json_configured': bool(GOOGLE_SERVICE_ACCOUNT_JSON),
        'drive_root_folder_id_configured': bool(GOOGLE_DRIVE_ROOT_FOLDER_ID)
    })


@app.route('/api/google/workspace/publish', methods=['POST'])
def publish_google_workspace():
    data = request.get_json() or {}
    item_type = data.get('item_type')
    item_id = data.get('item_id')
    class_name = data.get('class_name') or GOOGLE_WORKSPACE_DEFAULT_CLASS
    if not item_type or not item_id:
        return api_error(400, 'missing_item_identifiers', 'item_type and item_id are required')

    if item_type == 'exercise':
        item = get_or_404(Exercise, int(item_id), 'Exercise')
    elif item_type == 'exam':
        item = get_or_404(Exam, int(item_id), 'Exam')
    elif item_type == 'document':
        item = get_or_404(Document, int(item_id), 'Document')
    else:
        return api_error(400, 'unsupported_item_type', 'Unsupported item type')

    item_data = build_library_item(item_type, item)
    try:
        publish_data = publish_item_to_google_workspace(item_type, item_data, class_name=class_name)
    except Exception as exc:
        return api_error(400, 'google_workspace_publish_error', str(exc))

    audit_event('google.workspace.publish', {
        'item_type': item_type,
        'item_id': item_id,
        'class_name': class_name,
        'doc_id': publish_data.get('doc_id'),
        'folder_id': publish_data.get('folder_id')
    })
    return jsonify({
        'ok': True,
        'item_type': item_type,
        'item_id': item_id,
        **publish_data
    }), 201


@app.route('/api/google/workspace/publish-batch', methods=['POST'])
def publish_google_workspace_batch():
    if not feature_enabled('batch_publish'):
        return api_error(403, 'feature_disabled', 'batch_publish feature is disabled')

    data = request.get_json() or {}
    items = data.get('items') if isinstance(data.get('items'), list) else []
    class_name = data.get('class_name') or GOOGLE_WORKSPACE_DEFAULT_CLASS
    if not items:
        return api_error(400, 'missing_items', 'items list is required')
    if len(items) > 50:
        return api_error(400, 'batch_too_large', 'Maximum batch publish size is 50 items')

    published = []
    errors = []
    for idx, row in enumerate(items, start=1):
        try:
            item_type = row.get('item_type')
            item_id = row.get('item_id')
            if not item_type or not item_id:
                raise ValueError('missing_item_identifiers')
            if item_type == 'exercise':
                item = get_or_404(Exercise, int(item_id), 'Exercise')
            elif item_type == 'exam':
                item = get_or_404(Exam, int(item_id), 'Exam')
            elif item_type == 'document':
                item = get_or_404(Document, int(item_id), 'Document')
            else:
                raise ValueError('unsupported_item_type')
            item_data = build_library_item(item_type, item)
            publish_data = publish_item_to_google_workspace(item_type, item_data, class_name=class_name)
            published.append({
                'index': idx,
                'item_type': item_type,
                'item_id': int(item_id),
                **publish_data
            })
        except Exception as exc:
            errors.append({
                'index': idx,
                'item_type': row.get('item_type'),
                'item_id': row.get('item_id'),
                'error': str(exc)
            })

    audit_event('google.workspace.publish_batch', {
        'requested': len(items),
        'published': len(published),
        'errors': len(errors),
        'class_name': class_name
    })
    return jsonify({
        'ok': True,
        'requested': len(items),
        'published_count': len(published),
        'error_count': len(errors),
        'class_name': class_name,
        'published': published,
        'errors': errors
    }), 201


@app.route('/api/library/duplicate', methods=['POST'])
def duplicate_library_item():
    data = request.get_json() or {}
    item_type = data.get('item_type')
    item_id = data.get('item_id')
    if not item_type or not item_id:
        return api_error(400, 'missing_item_identifiers', 'item_type and item_id are required')

    if item_type == 'exercise':
        src = get_or_404(Exercise, int(item_id), 'Exercise')
        clone = Exercise(
            title=f"{src.title} (copia)",
            topic=src.topic,
            level=src.level,
            exercise_type=src.exercise_type,
            content=src.content,
            ai_model=src.ai_model
        )
        db.session.add(clone)
        db.session.commit()
        return jsonify(build_library_item('exercise', clone)), 201
    if item_type == 'exam':
        src = get_or_404(Exam, int(item_id), 'Exam')
        clone = Exam(
            title=f"{src.title} (copia)",
            description=src.description,
            exercises=src.exercises,
            total_score=src.total_score
        )
        db.session.add(clone)
        db.session.commit()
        return jsonify(build_library_item('exam', clone)), 201
    if item_type == 'document':
        src = get_or_404(Document, int(item_id), 'Document')
        clone = Document(
            filename=f"{src.filename} (copia)",
            analysis=src.analysis
        )
        db.session.add(clone)
        db.session.commit()
        return jsonify(build_library_item('document', clone)), 201
    return api_error(400, 'unsupported_item_type', 'Unsupported item type')


@app.route('/api/library/export/moodle-xml', methods=['POST'])
def export_library_moodle_xml():
    data = request.get_json() or {}
    items = data.get('items') or []
    options = data.get('options') or {}
    if not isinstance(items, list) or len(items) == 0:
        return api_error(400, 'missing_items', 'items list is required')
    include_answers = bool(options.get('include_answers', True))
    resolved = []
    for item in items:
        item_type = item.get('type')
        item_id = item.get('id')
        if not item_type or not item_id:
            continue
        model = get_or_404(Exercise if item_type == 'exercise' else Exam if item_type == 'exam' else Document, int(item_id), item_type.title())
        resolved.append((item_type, build_library_item(item_type, model)))
    xml_content = build_moodle_xml(resolved, include_answers=include_answers)
    exports_dir = get_exports_dir()
    out_file = exports_dir / f"moodle_export_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.xml"
    out_file.write_text(xml_content, encoding='utf-8')
    token = register_download_file(out_file)
    audit_event('library.export.moodle_xml', {'items': len(resolved)})
    return jsonify({'ok': True, 'file_path': str(out_file), 'download_name': out_file.name, 'download_url': f'/api/library/export/download/{token}'}), 201


@app.route('/api/library/export/h5p-json', methods=['POST'])
def export_library_h5p_json():
    data = request.get_json() or {}
    items = data.get('items') or []
    options = data.get('options') or {}
    if not isinstance(items, list) or len(items) == 0:
        return api_error(400, 'missing_items', 'items list is required')
    include_answers = bool(options.get('include_answers', True))
    resolved = []
    for item in items:
        item_type = item.get('type')
        item_id = item.get('id')
        if not item_type or not item_id:
            continue
        model = get_or_404(Exercise if item_type == 'exercise' else Exam if item_type == 'exam' else Document, int(item_id), item_type.title())
        resolved.append((item_type, build_library_item(item_type, model)))
    payload = build_h5p_payload(resolved, include_answers=include_answers)
    exports_dir = get_exports_dir()
    out_file = exports_dir / f"h5p_export_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    token = register_download_file(out_file)
    audit_event('library.export.h5p_json', {'items': len(resolved)})
    return jsonify({'ok': True, 'file_path': str(out_file), 'download_name': out_file.name, 'download_url': f'/api/library/export/download/{token}'}), 201


@app.route('/api/library/export/notebooklm-pack', methods=['POST'])
def export_library_notebooklm_pack():
    data = request.get_json() or {}
    items = data.get('items') or []
    if not isinstance(items, list) or len(items) == 0:
        return api_error(400, 'missing_items', 'items list is required')
    exports_dir = get_exports_dir()
    timestamp = datetime.now(UTC).strftime('%Y%m%d_%H%M%S')
    zip_path = exports_dir / f'notebooklm_pack_{timestamp}.zip'

    with ZipFile(zip_path, 'w', ZIP_DEFLATED) as zf:
        guide_lines = ['Pack NotebookLM - French Exercise App', '', 'Contenido:']
        for item in items:
            item_type = item.get('type')
            item_id = item.get('id')
            if not item_type or not item_id:
                continue
            model = get_or_404(Exercise if item_type == 'exercise' else Exam if item_type == 'exam' else Document, int(item_id), item_type.title())
            item_data = build_library_item(item_type, model)
            file_stub = f'{item_type}_{item_id}'
            zf.writestr(f'{file_stub}.json', json.dumps(item_data, ensure_ascii=False, indent=2))
            zf.writestr(f'{file_stub}.txt', json.dumps(item_data, ensure_ascii=False, indent=2))
            guide_lines.append(f'- {file_stub}: {item_data.get("display_title")}')
        zf.writestr('README.txt', '\n'.join(guide_lines))

    token = register_download_file(zip_path)
    audit_event('library.export.notebooklm_pack', {'items': len(items)})
    return jsonify({'ok': True, 'file_path': str(zip_path), 'download_name': zip_path.name, 'download_url': f'/api/library/export/download/{token}'}), 201


@app.route('/api/library/export/download/<token>', methods=['GET'])
def download_export_file(token):
    from flask import send_file
    file_path = EXPORT_TOKEN_MAP.get(token)
    if not file_path:
        return api_error(404, 'invalid_download_token', 'Invalid or expired download token')
    target = Path(file_path)
    if not target.exists():
        return api_error(404, 'file_not_found', 'File not found')
    return send_file(target, as_attachment=True, download_name=target.name)


@app.route('/api/library/open', methods=['POST'])
def open_library_path():
    data = request.get_json() or {}
    path = data.get('path')
    if not path:
        return api_error(400, 'missing_path', 'path is required')

    target = Path(path).resolve()
    exports_dir = get_exports_dir().resolve()
    # Safety: allow opening only inside exports directory.
    if exports_dir not in target.parents and target != exports_dir:
        return api_error(403, 'path_not_allowed', 'Path outside exports directory is not allowed')

    try:
        subprocess.run(['open', str(target)], check=True)
    except Exception as e:
        return api_error(500, 'open_path_failed', f'Failed to open path: {e}')

    return jsonify({'opened': str(target)})


@app.route('/api/compliance/status', methods=['GET'])
def compliance_status():
    return jsonify({
        'mode': 'canarias' if CANARIAS_COMPLIANCE_MODE else 'standard',
        'pii_filter': CANARIAS_COMPLIANCE_MODE,
        'ai_grading_blocked': True
    })


@app.route('/api/compliance/anonymize-preview', methods=['POST'])
def compliance_anonymize_preview():
    data = request.get_json() or {}
    text = data.get('text', '')
    if not isinstance(text, str):
        return api_error(400, 'invalid_text_type', 'text must be a string')
    sanitized, detections = sanitize_for_cloud_prompt(text)
    audit_event('compliance.anonymize_preview', {'detections': detections})
    return jsonify({'sanitized': sanitized, 'detections': detections})


@app.route('/api/compliance/audit-log', methods=['GET'])
def compliance_audit_log():
    from_dt = request.args.get('from')
    to_dt = request.args.get('to')
    action_filter = request.args.get('action')
    fmt = request.args.get('format')
    # advanced search
    q = request.args.get('q')
    q_regex = request.args.get('q_regex')
    fields_param = request.args.get('fields')
    fields = [f.strip() for f in fields_param.split(',')] if fields_param else None
    # pagination params
    try:
        limit = int(request.args.get('limit', '100'))
        if limit < 0:
            raise ValueError
    except ValueError:
        return api_error(400, 'invalid_limit', 'limit must be a non-negative integer')
    try:
        offset = int(request.args.get('offset', '0'))
        if offset < 0:
            raise ValueError
    except ValueError:
        return api_error(400, 'invalid_offset', 'offset must be a non-negative integer')

    # increment query metric
    try:
        METRICS['audit_log_queries'] = METRICS.get('audit_log_queries', 0) + 1
    except Exception:
        pass

    entries = list(AUDIT_LOG)

    filtered = []
    for row in entries:
        ts = row.get('timestamp')
        try:
            ts_dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        except Exception:
            continue
        include = True
        if from_dt:
            try:
                include = include and ts_dt >= datetime.fromisoformat(from_dt)
            except Exception:
                pass
        if to_dt:
            try:
                include = include and ts_dt <= datetime.fromisoformat(to_dt)
            except Exception:
                pass
        if action_filter and isinstance(row.get('action'), str):
            include = include and action_filter.lower() in row['action'].lower()
        # advanced q / q_regex across fields
        if include and (q or q_regex):
            matched = False
            target_fields = fields or ['action', 'detail']
            for fld in target_fields:
                try:
                    # support nested like detail.provider
                    parts = fld.split('.')
                    val = row
                    for p in parts:
                        if isinstance(val, dict):
                            val = val.get(p)
                        else:
                            val = None
                            break
                    text = '' if val is None else (json.dumps(val) if not isinstance(val, (str, int, float)) else str(val))
                except Exception:
                    text = ''
                if q_regex:
                    try:
                        if re.search(q_regex, text, re.IGNORECASE):
                            matched = True
                            break
                    except re.error:
                        pass
                if q and q.lower() in text.lower():
                    matched = True
                    break
            include = include and matched
        if include:
            filtered.append(row)

    total = len(filtered)
    # apply pagination
    paged = filtered[offset: offset + limit] if limit is not None else filtered[offset:]

    # support CSV export
    if fmt == 'csv':
        si = io.StringIO()
        writer = csv.writer(si)
        writer.writerow(['timestamp', 'action', 'detail'])
        for r in filtered:
            writer.writerow([
                r.get('timestamp'),
                r.get('action'),
                json.dumps(r.get('detail')) if r.get('detail') is not None else ''
            ])
        output = si.getvalue()
        return Response(
            output,
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment; filename="audit_log.csv"'}
        )

    # support JSON export of all filtered entries ignoring pagination
    if fmt == 'json':
        return Response(
            json.dumps(filtered, ensure_ascii=False),
            mimetype='application/json',
            headers={'Content-Disposition': 'attachment; filename="audit_log.json"'}
        )

    # support XLSX export of all filtered entries
    if fmt == 'xlsx' or fmt == 'xls' or fmt == 'excel':
        if Workbook is None:
            return api_error(500, 'missing_dependency', 'openpyxl is not installed')
        wb = Workbook()
        ws = wb.active
        ws.append(['timestamp', 'action', 'detail'])
        for r in filtered:
            ws.append([
                r.get('timestamp'),
                r.get('action'),
                json.dumps(r.get('detail')) if r.get('detail') is not None else ''
            ])
        bio = io.BytesIO()
        wb.save(bio)
        bio.seek(0)
        return Response(
            bio.read(),
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            headers={'Content-Disposition': 'attachment; filename="audit_log.xlsx"'}
        )

    # default response includes pagination metadata
    return jsonify({
        'total': total,
        'limit': limit,
        'offset': offset,
        'entries': paged
    })


@app.route('/api/exercises/templates', methods=['GET'])
def get_exercise_templates():
    templates = {
        'themes': ['les couleurs', 'au téléphone', 'les vêtements', 'objets du quotidien', 'animales', 'école'],
        'activity_types': MAGIC_ACTIVITY_TYPES,
        'default_mode': 'magic_mix',
        'constraints': {'min_variety': 3},
        'template_engine': PEDAGOGICAL_TEMPLATE_ENGINE,
        'quality': {
            'single_min_score': QUALITY_MIN_SCORE,
            'batch_min_score': QUALITY_BATCH_MIN_SCORE
        }
    }
    return jsonify(templates)


@app.route('/api/assistant/create-exercise', methods=['POST'])
def assistant_create_exercise():
    """Generador simple y creativo de ejercicios para docentes.
    Recibe JSON con parámetros opcionales:
      - level: 'A1'|'A2'|'B1' etc. (default 'A1')
      - type: 'fill_blank'|'multiple_choice'|'translate'|'role_play'|'conjugation'|'matching' (default 'fill_blank')
      - count: número de ejercicios a generar (default 1)
      - topics: lista de temas sugeridos
      - style: 'creative'|'concise' (afecta variaciones)
      - seed: opcional para reproducibilidad
    Devuelve JSON con `generated` y `suggestions`.
    """
    payload = request.get_json() or {}
    level = str(payload.get('level') or 'A1')
    activity = str(payload.get('type') or 'fill_blank')
    try:
        count = max(1, int(payload.get('count', 1)))
    except Exception:
        count = 1
    topics = payload.get('topics') if isinstance(payload.get('topics'), list) else [payload.get('topic') or 'les couleurs']
    style = str(payload.get('style') or 'creative')
    seed = payload.get('seed')
    if seed is not None:
        try:
            random.seed(int(seed))
        except Exception:
            random.seed()

    # pequeñas bases de contenido
    COLORS = ['rouge', 'bleu', 'vert', 'jaune', 'noir', 'blanc']
    CLOTHES = ['chemise', 'pantalon', 'robe', 'chaussures', 'chapeau']
    ANIMALS = ['chat', 'chien', 'cheval', 'oiseau', 'poisson']
    OBJECTS = ['livre', 'stylo', 'téléphone', 'chaise', 'table']

    def mk_id():
        return str(uuid.uuid4())

    def gen_fill_blank(t):
        topic = t or random.choice(topics)
        if 'couleur' in topic or 'couleur' in topic.lower():
            word = random.choice(COLORS)
            prompt = f"J'aime la couleur ____ sur ma voiture. Complète avec le mot correct (niveau {level})."
            solution = word
        elif 'vêt' in topic or 'vêt' in topic.lower():
            word = random.choice(CLOTHES)
            prompt = f"Je porte une ____ aujourd'hui. Complète la phrase."
            solution = word
        else:
            word = random.choice(OBJECTS)
            prompt = f"Où est mon ____ ? Complète avec le mot approprié."
            solution = word
        return {'id': mk_id(), 'type': 'fill_blank', 'prompt': prompt, 'solution': solution, 'level': level, 'topic': topic}

    def gen_multiple_choice(t):
        topic = t or random.choice(topics)
        word = random.choice(ANIMALS + COLORS + OBJECTS)
        distractors = random.sample([w for w in (ANIMALS + COLORS + OBJECTS) if w != word], 3)
        choices = distractors + [word]
        random.shuffle(choices)
        prompt = f"Choisis le mot correct pour compléter: 'Le garçon a un {word}.' (elije la mejor opción)"
        return {'id': mk_id(), 'type': 'multiple_choice', 'prompt': prompt, 'choices': choices, 'solution': word, 'level': level, 'topic': topic}

    def gen_translate(t):
        topic = t or random.choice(topics)
        src = random.choice(['I have a book.', 'The cat is sleeping.', 'She wears a red dress.'])
        prompt = f"Traduisez en français (nivel {level}): '{src}'"
        # Not attempting full auto-translation; provide simple expected answers
        simple_map = {
            'I have a book.': 'J\'ai un livre.',
            'The cat is sleeping.': 'Le chat dort.',
            'She wears a red dress.': 'Elle porte une robe rouge.'
        }
        return {'id': mk_id(), 'type': 'translate', 'prompt': prompt, 'solution': simple_map.get(src), 'level': level, 'topic': topic}

    def gen_role_play(t):
        topic = t or random.choice(topics)
        roles = ['élève', 'enseignant', 'vendeur', 'client']
        situation = random.choice(['au marché', 'au téléphone', 'au restaurant', 'à l\'école'])
        prompt = f"Role-play: Jouez le dialogue entre {roles[0]} et {roles[1]} {situation}. Fournissez 6 lignes de dialogue adaptées au niveau {level}."
        return {'id': mk_id(), 'type': 'role_play', 'prompt': prompt, 'notes': 'Esperar respuesta libre; el docente puede adaptar el rol.', 'level': level, 'topic': topic}

    def gen_conjugation(t):
        topic = t or random.choice(topics)
        verb = random.choice(['aller', 'être', 'avoir', 'faire'])
        tense = random.choice(['présent', 'passé composé', 'futur proche'])
        prompt = f"Conjuguez le verbe '{verb}' au {tense} pour 'nous' et 'ils' (nivel {level})."
        return {'id': mk_id(), 'type': 'conjugation', 'prompt': prompt, 'solution': None, 'level': level, 'topic': topic}

    def gen_matching(t):
        topic = t or random.choice(topics)
        left = ['chien', 'chat', 'oiseau']
        right = ['dog', 'cat', 'bird']
        random.shuffle(right)
        prompt = f"Associez les mots français à leur équivalent anglais (niveau {level})."
        return {'id': mk_id(), 'type': 'matching', 'prompt': prompt, 'pairs': list(zip(left, right)), 'level': level, 'topic': topic}

    generators = {
        'fill_blank': gen_fill_blank,
        'multiple_choice': gen_multiple_choice,
        'translate': gen_translate,
        'role_play': gen_role_play,
        'conjugation': gen_conjugation,
        'matching': gen_matching
    }

    generated = []
    for i in range(count):
        t = activity if activity in generators else random.choice(list(generators.keys()))
        gen = generators[t]
        topic_choice = topics[i % len(topics)] if topics else None
        try:
            item = gen(topic_choice)
        except Exception:
            item = gen(None)
        generated.append(item)

    # propuestas proactivas
    suggestions = [
        {'tip': 'Crear una versión multiple-choice a partir de un fill_blank.'},
        {'tip': 'Generar 3 variantes con distinto nivel de retroalimentación.'},
        {'tip': 'Añadir clave de corrección y rúbrica para la auto-evaluación.'}
    ]

    # registrar en el log de auditoría
    try:
        audit_event('assistant.generate', {'count': len(generated), 'type': activity, 'level': level})
    except Exception:
        pass

    return jsonify({'ok': True, 'generated': generated, 'suggestions': suggestions})


@app.route('/api/exercises/quality/evaluate-batch', methods=['POST'])
def evaluate_exercises_quality_batch():
    data = request.get_json() or {}
    exercise_ids = data.get('exercise_ids') if isinstance(data.get('exercise_ids'), list) else []
    if not exercise_ids:
        return api_error(400, 'missing_exercise_ids', 'exercise_ids is required')
    if len(exercise_ids) > 200:
        return api_error(400, 'batch_too_large', 'Maximum 200 exercises per quality evaluation')

    reports = []
    for raw_id in exercise_ids:
        try:
            exercise_id = int(raw_id)
        except Exception:
            reports.append({'id': raw_id, 'ok': False, 'error': 'invalid_id'})
            continue

        exercise = db.session.get(Exercise, exercise_id)
        if not exercise:
            reports.append({'id': exercise_id, 'ok': False, 'error': 'not_found'})
            continue

        expected_type = exercise.exercise_type if exercise.exercise_type in MAGIC_ACTIVITY_TYPES else None
        quality = evaluate_exercise_quality(exercise.content or {}, expected_type=expected_type)
        reports.append({
            'id': exercise_id,
            'ok': True,
            'title': exercise.title,
            'activity_type': (exercise.content or {}).get('activity_type', exercise.exercise_type),
            'quality': quality
        })

    passed_count = sum(1 for row in reports if row.get('ok') and row.get('quality', {}).get('passed'))
    return jsonify({
        'ok': True,
        'evaluated': len(reports),
        'passed': passed_count,
        'failed': len(reports) - passed_count,
        'threshold': QUALITY_MIN_SCORE,
        'reports': reports
    })


@app.route('/api/exercises/repair-batch', methods=['POST'])
def repair_batch_exercises():
    return jsonify(_run_repair_exercises())


@app.route('/api/enterprise/features', methods=['GET'])
def enterprise_features():
    return jsonify({
        'features': sorted(list(ENTERPRISE_FEATURES)),
        'count': len(ENTERPRISE_FEATURES)
    })


@app.route('/api/analytics/learning', methods=['GET'])
def analytics_learning():
    exercises = Exercise.query.all()
    by_type = {}
    by_topic = {}
    for item in exercises:
        by_type[item.exercise_type] = by_type.get(item.exercise_type, 0) + 1
        by_topic[item.topic] = by_topic.get(item.topic, 0) + 1
    return jsonify({
        'totals': {
            'exercises': len(exercises),
            'documents': Document.query.count(),
            'exams': Exam.query.count()
        },
        'exercise_by_type': by_type,
        'exercise_by_topic': by_topic,
        'library_reuse_estimate': {
            'duplicated_items': sum(1 for e in exercises if '(copia)' in (e.title or ''))
        }
    })


@app.route('/api/ops/metrics', methods=['GET'])
def ops_metrics():
    if not feature_enabled('ops_metrics'):
        return api_error(403, 'feature_disabled', 'ops_metrics feature is disabled')
    now = datetime.now(UTC)
    window_24h = now - timedelta(hours=24)
    exercises = Exercise.query.all()
    exams = Exam.query.all()
    documents = Document.query.all()
    chats = ChatMessage.query.all()
    recent_exercises = [e for e in exercises if as_utc(e.created_at) and as_utc(e.created_at) >= window_24h]
    by_type = {}
    for e in exercises:
        key = (e.content or {}).get('activity_type', e.exercise_type)
        by_type[key] = by_type.get(key, 0) + 1
    return jsonify({
        'timestamp': now.isoformat(),
        'totals': {
            'exercises': len(exercises),
            'exams': len(exams),
            'documents': len(documents),
            'chat_messages': len(chats),
            'audit_events': len(AUDIT_LOG)
        },
        'window_24h': {
            'exercises_created': len(recent_exercises)
        },
        'exercise_by_type': by_type,
        'ai_keys': ai_keys_health_snapshot().get('summary', {})
    })

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        auto_restore_if_empty()
        key_health = ai_keys_health_snapshot()
        log_event(
            logging.INFO,
            'startup.ai_keys_health',
            status=key_health['summary']['status'],
            providers_ok=key_health['summary']['providers_ok'],
            providers_total=key_health['summary']['providers_total'],
            missing_vars=key_health['missing_vars']
        )
    # Avoid duplicate scheduler instances when Flask debug reloader spawns a second process.
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true' or not DEBUG_MODE:
        start_backup_scheduler()
    app.run(debug=DEBUG_MODE, host=HOST, port=PORT)
