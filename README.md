# French Exercise App - Generador de Ejercicios de Francés

## Descripción

Aplicación web interactiva y PWA para maestras de francés en primaria que genera ejercicios personalizados con IA local y en nube. Incluye creación de exámenes, escaneo de documentos y análisis con IA.

## Características

- Generación automática de ejercicios (A1-A2)
- Creación de exámenes de evaluación
- Escaneo de documentos con cámara móvil
- Análisis de IA sobre documentos y respuestas
- Responsive para iPhone, iPad y tablets
- PWA (funciona offline)
- IA híbrida (local + nube)
- Base de datos con persistencia

## Tech Stack

**Backend:** Flask, Python 3.10+, SQLAlchemy, SQLite
**Frontend:** React 18, Vite, Tailwind CSS, PWA
**IA:** Ollama (local), Perplexity, OpenAI, Gemini, DeepSeek (nube)

## Instalación Rápida

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

Accede a http://localhost:5173

## Estructura

```
french-exercise-app/
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   ├── models/
│   ├── services/
│   └── routes/
├── frontend/
│   ├── src/
│   ├── public/
│   └── package.json
└── README.md
```

## API Endpoints

- POST /api/exercises/generate - Generar ejercicio
- GET /api/exercises - Listar ejercicios
- POST /api/exams - Crear examen
- POST /api/documents/upload - Subir documento
- POST /api/documents/analyze - Analizar documento

## Licencia

MIT
