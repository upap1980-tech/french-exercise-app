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


## Instalación y Configuración

### Requisitos Previos
- Python 3.10+
- Node.js 16+
- npm o yarn
- Ollama (para IA local) - Descargar desde https://ollama.ai
- Claves de API para Perplexity, Google Gemini, OpenAI, o DeepSeek

### Backend - Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edita .env con tus claves de API
python app.py
```

El backend estará disponible en http://localhost:5007

### Frontend - Setup

```bash
cd frontend
npm install
npm run dev
```

La aplicación estará disponible en http://localhost:3000

### Configuración de Ollama (Local AI)

```bash
# Descargar modelo Llama 3
ollama pull llama2

# Iniciar Ollama
ollama serve
```

Ollama estará disponible en http://localhost:11434

## Uso de la Aplicación

### Panel de Control
- Visualiza estadísticas de ejercicios generados
- Ve los modelos IA disponibles
- Accede a las características principales

### Generar Ejercicios
1. Ingresa un tema (Ej: "Verbos en presente", "Vocabulario de frutas")
2. Selecciona la dificultad (Fácil, Intermedio, Difícil)
3. Elige el modelo IA (Ollama local o servicios en la nube)
4. Haz clic en "Generar Ejercicios"

### Subir Documentos
1. Selecciona un documento (PDF, JPG, PNG, DOC, DOCX)
2. Haz clic en "Subir y Analizar"
3. La IA analizará el contenido y proporcionará resultados

## Despliegue en Producción

### Frontend (Vercel o Netlify)

```bash
cd frontend
npm run build
```

Sube la carpeta `dist/` a Vercel o Netlify

### Backend (Heroku, Railway, o VPS)

```bash
cd backend
echo "web: python app.py" > Procfile
git push
```

## Características Principales ✨

✅ Generación inteligente de ejercicios con IA
✅ Soporte para múltiples modelos IA (local + nube)
✅ Análisis de documentos escaneados
✅ Generación de exámenes de evaluación
✅ Interfaz responsiva para iPhone y tablets
✅ PWA para instalación en dispositivos
✅ Trabajar offline

## Tecnologías Utilizadas

**Backend:**
- Flask (Python web framework)
- SQLAlchemy (ORM)
- Ollama (IA Local)
- Perplexity API
- Google Gemini API
- OpenAI API

**Frontend:**
- React 18
- Vite (build tool)
- Responsive CSS3
- PWA manifest

## Contribuir

Las contribuciones son bienvenidas. Por favor:
1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Haz commit de tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## Contacto

Víctor - [@upap1980_tech](https://github.com/upap1980-tech)

Proyecto: [French Exercise App](https://github.com/upap1980-tech/french-exercise-app)
