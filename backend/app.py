import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
import requests
import json
from datetime import datetime
import base64
oad_dotenv()

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

db = SQLAlchemy(app)
CORS(app)

# Configuración de IA
OLLAMA_URL = os.getenv('OLLAMA_API_URL', 'http://localhost:11434')
PERPLEXITY_KEY = os.getenv('PERPLEXITY_API_KEY', '')
OPENAI_KEY = os.getenv('OPENAI_API_KEY', '')
GEMINI_KEY = os.getenv('GEMINI_API_KEY', '')
DEEPSEEK_KEY = os.getenv('DEEPSEEK_API_KEY', '')

class Exercise(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    topic = db.Column(db.String(100), nullable=False)
    level = db.Column(db.String(10), nullable=False)
    exercise_type = db.Column(db.String(50), nullable=False)
    content = db.Column(db.JSON, nullable=False)
    ai_model = db.Column(db.String(50), default='local')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'topic': self.topic,
            'level': self.level,
            'exercise_type': self.exercise_type,
            'content': self.content,
            'ai_model': self.ai_model,
            'created_at': self.created_at.isoformat()
        }

class Exam(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    exercises = db.Column(db.JSON, nullable=False)
    total_score = db.Column(db.Integer, default=100)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'exercises': self.exercises,
            'total_score': self.total_score,
            'created_at': self.created_at.isoformat()
        }

class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    file_data = db.Column(db.LargeBinary)
    analysis = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.filename,
            'analysis': self.analysis,
            'created_at': self.created_at.isoformat()
        }

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
        print(f'Ollama error: {e}')
    return None

def call_perplexity(prompt):
    try:
        if not PERPLEXITY_KEY:
            return None
        headers = {'Authorization': f'Bearer {PERPLEXITY_KEY}'}
        data = {'model': 'pplx-70b-online', 'messages': [{'role': 'user', 'content': prompt}]}
        response = requests.post('https://api.perplexity.ai/chat/completions', json=data, headers=headers)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f'Perplexity error: {e}')
    return None

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'French Exercise App Backend is running'})

@app.route('/api/exercises', methods=['GET'])
def get_exercises():
    exercises = Exercise.query.all()
    return jsonify([e.to_dict() for e in exercises])

@app.route('/api/exercises/generate', methods=['POST'])
def generate_exercise():
    data = request.get_json()
    exercise_type = data.get('exercise_type', 'fill_blank')
    topic = data.get('topic', 'animals')
    level = data.get('level', 'A1')
    ai_mode = data.get('ai_mode', 'local')
    
    prompt = f'''Generate a French {exercise_type} exercise for {level} level about {topic}.
    Format the response as JSON with fields: question, options (if multiple choice), correct_answer, explanation.
    Make it appropriate for primary school students.'''
    
    if ai_mode == 'cloud' and PERPLEXITY_KEY:
        content = call_perplexity(prompt)
    else:
        content = call_ollama(prompt)
    
    if not content:
        return jsonify({'error': 'Failed to generate exercise'}), 500
    
    try:
        exercise_data = json.loads(content) if '{' in content else {'question': content}
    except:
        exercise_data = {'question': content}
    
    exercise = Exercise(
        title=f"{topic.capitalize()} - {exercise_type.replace('_', ' ').title()}",
        topic=topic,
        level=level,
        exercise_type=exercise_type,
        content=exercise_data,
        ai_model=ai_mode
    )
    db.session.add(exercise)
    db.session.commit()
    
    return jsonify(exercise.to_dict()), 201

@app.route('/api/exercises/<int:id>', methods=['GET'])
def get_exercise(id):
    exercise = Exercise.query.get_or_404(id)
    return jsonify(exercise.to_dict())

@app.route('/api/exercises/<int:id>', methods=['DELETE'])
def delete_exercise(id):
    exercise = Exercise.query.get_or_404(id)
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
    exam = Exam.query.get_or_404(id)
    return jsonify(exam.to_dict())

@app.route('/api/exams/<int:id>', methods=['DELETE'])
def delete_exam(id):
    exam = Exam.query.get_or_404(id)
    db.session.delete(exam)
    db.session.commit()
    return jsonify({'message': 'Exam deleted'})

@app.route('/api/documents/upload', methods=['POST'])
def upload_document():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    file_data = file.read()
    document = Document(filename=file.filename, file_data=file_data)
    db.session.add(document)
    db.session.commit()
    
    return jsonify(document.to_dict()), 201

@app.route('/api/documents/<int:id>/analyze', methods=['POST'])
def analyze_document(id):
    document = Document.query.get_or_404(id)
    
    ai_mode = request.get_json().get('ai_mode', 'local') if request.is_json else 'local'
    
    prompt = f'''Analyze this document and extract:
    1. Main topic
    2. Key information
    3. Vocabulary (if in French)
    4. Suggested exercises based on content
    
    Respond in JSON format.'''
    
    if ai_mode == 'cloud' and PERPLEXITY_KEY:
        analysis = call_perplexity(prompt)
    else:
        analysis = call_ollama(prompt)
    
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
        'cloud': []
    }
    if PERPLEXITY_KEY:
        models['cloud'].append('perplexity')
    if OPENAI_KEY:
        models['cloud'].append('openai')
    if GEMINI_KEY:
        models['cloud'].append('gemini')
    if DEEPSEEK_KEY:
        models['cloud'].append('deepseek')
    
    return jsonify(models)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5007)
