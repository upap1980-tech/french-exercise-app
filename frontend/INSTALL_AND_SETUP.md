# Guía de Instalación y Configuración - French Exercise App

## Instalación Rápida

### Requisitos Previos
- Python 3.10+
- Node.js 18+
- Git
- Ollama (para IA local)

### Paso 1: Clonar el Repositorio

```bash
git clone https://github.com/upap1980-tech/french-exercise-app.git
cd french-exercise-app
```

### Paso 2: Configurar Backend

```bash
cd backend

# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Crear archivo .env
cp .env.example .env

# Editar .env con tus API keys (opcional, funciona sin ellas con Ollama local)
```

### Paso 3: Instalar y Configurar Ollama (Opcional pero Recomendado)

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Descargar modelo Llama3
ollama pull llama3

# Ejecutar Ollama en otra terminal
ollama serve
```

### Paso 4: Ejecutar Backend

En la carpeta `backend`:

```bash
python app.py
```

El backend estará disponible en `http://localhost:5000`

### Paso 5: Configurar Frontend

En una nueva terminal, desde la raíz:

```bash
cd frontend

# Instalar dependencias
npm install

# Ejecutar servidor de desarrollo
npm run dev
```

El frontend estará disponible en `http://localhost:5173`

## Uso de la Aplicación

### Generar un Ejercicio

1. Ve a la sección "Generar Ejercicio" en el dashboard
2. Selecciona:
   - Tipo de ejercicio (Completar frases, Conjugación, etc.)
   - Tema (Animales, Familia, Verbos, etc.)
   - Nivel (A1, A2)
   - Modo IA (Local con Ollama o Nube con Perplexity/OpenAI)
3. Haz clic en "Generar"
4. Previsualiza y guarda el ejercicio

### Crear un Exámen

1. Ve a "Crear Exámen"
2. Añade múltiples ejercicios
3. Configura el título y descripción
4. Define la puntuación total
5. Guarda y comparte

### Escanear Documentos (Móvil)

1. Abre la app en iPhone/iPad
2. Ve a "Escanear Documento"
3. Otorga acceso a la cámara
4. Escanea el documento
5. La IA analizará el contenido automáticamente

## Variables de Entorno (.env)

Copia el contenido de `.env.example` a `.env` y personaliza:

```env
# Backend
FLASK_ENV=development
FLASK_SECRET_KEY=tu-clave-secreta
DATABASE_URL=sqlite:///database.db

# IA Local (Requerido para funcionar sin APIs nube)
OLLAMA_API_URL=http://localhost:11434
OLLAMA_MODEL=llama3

# APIs Opcionales (para IA en nube)
PERPLEXITY_API_KEY=tu-api-key
OPENAI_API_KEY=tu-api-key
GEMINI_API_KEY=tu-api-key
DEEPSEEK_API_KEY=tu-api-key

# Frontend
VITE_API_URL=http://localhost:5000
```

## Estructura de Carpetas

```
french-exercise-app/
├── backend/
│   ├── app.py                 # Aplicación Flask principal
│   ├── requirements.txt       # Dependencias Python
│   ├── .env.example           # Plantilla de configuración
│   └── database.db            # Base de datos SQLite
├── frontend/
│   ├── src/
│   │   ├── App.jsx                # Componente principal
│   │   ├── main.jsx               # Punto de entrada
│   │   ├── components/            # Componentes React
│   │   └── services/              # Servicios API
│   ├── public/
│   │   ├── manifest.json          # PWA manifest
│   │   └── service-worker.js      # Service Worker
│   ├── package.json           # Dependencias Node
│   └── vite.config.js         # Configuración Vite
├── .gitignore
├── README.md              # Documentación
└── INSTALL_AND_SETUP.md   # Esta guía
```

## Troubleshooting

### Problema: Ollama no conecta
**Solución:**
```bash
# En otra terminal, ejecuta:
ollama serve
```

### Problema: Puerto 5000 ya está en uso
**Solución:**
```bash
# Usa un puerto diferente:
python app.py --port 5001
# Y actualiza VITE_API_URL en .env
```

### Problema: Puerto 5173 ya está en uso
**Solución:**
```bash
# Vite usará automáticamente el siguiente puerto disponible
```

### Problema: CORS errors
**Solución:**
Verifica que:
1. El backend está corriendo en `http://localhost:5000`
2. El frontend tiene configurado `VITE_API_URL=http://localhost:5000` en .env

### Problema: No puedo generar ejercicios
**Causas posibles:**
1. Ollama no está corriendo (si usas IA local)
2. No tienes API keys configuradas (si usas IA nube)
3. El modelo llama3 no está descargado

## API Endpoints Disponibles

### Ejercicios
- `GET /api/exercises` - Listar todos los ejercicios
- `POST /api/exercises/generate` - Generar nuevo ejercicio
- `GET /api/exercises/<id>` - Obtener un ejercicio
- `DELETE /api/exercises/<id>` - Eliminar un ejercicio

### Exámenes
- `GET /api/exams` - Listar todos los exámenes
- `POST /api/exams` - Crear nuevo examen
- `GET /api/exams/<id>` - Obtener un examen
- `DELETE /api/exams/<id>` - Eliminar un examen

### Documentos
- `POST /api/documents/upload` - Subir documento
- `POST /api/documents/<id>/analyze` - Analizar documento

### IA
- `GET /api/ai/models` - Listar modelos IA disponibles

## Desplegar en Producción

### Backend (Heroku/Railway)

1. Crear archivo `Procfile` en backend:
   ```
   web: gunicorn app:app
   ```

2. Desplegar con Git

### Frontend (Vercel/Netlify)

```bash
cd frontend
npm run build
# Sube la carpeta 'dist' a tu servicio de hosting
```

## Licencia

MIT - Ver LICENSE.md para más detalles

## Soporte

Para reportar problemas o sugerencias, crea un issue en GitHub:
https://github.com/upap1980-tech/french-exercise-app/issues
