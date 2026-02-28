# Análisis de Materiales Docentes (FRANÇAIS 6º)

Ruta analizada:
`/Volumes/BEA/CURSO 2025-2026/FRANÇAIS 6º`

## Resumen rápido

- Total archivos detectados (excluyendo `._`): **58**
- Extensiones:
  - `.pdf`: 32
  - `.odt`: 20
  - `.ods`: 2
  - `.mp4`: 2
  - `.odp`: 1
  - `.jpg`: 1

## Bloques temáticos útiles para enriquecer la app

1. `heures_temps` (13 archivos)
- Ejemplos:
  - `U. 1 MES ACTIVITÉS/activités sur l'emploi du temps et les heures.pdf`
  - `U. 1 MES ACTIVITÉS/Fiche l'horloge.pdf`
  - `U. 1 MES ACTIVITÉS/ACT Choisis l'heure correspondante..pdf`

2. `vêtements` (15 archivos)
- Ejemplos:
  - `U.6 DANS MON ARMOIRE/VOC LES VETEMENTS.pdf`
  - `U.6 DANS MON ARMOIRE/VOCABULAIRE ILLUSTRÉ LES VÊTEMENTS.pdf`
  - `U.6 DANS MON ARMOIRE/Qu'est-ce qu'il- elle porte - Images-.jpg`

3. `corps_description` (11 archivos)
- Ejemplos:
  - `U.2 MES SUPER HÉROS - LE CORPS- DESCRIPTION/crucigrama Les parties du corps.pdf`
  - `U.2 MES SUPER HÉROS - LE CORPS- DESCRIPTION/ÉCRIS LES PARTIES DU CORPS.pdf`
  - `U.2 MES SUPER HÉROS - LE CORPS- DESCRIPTION/teoria AVOIR MAL À ....pdf`

4. `france_culture` (7 archivos)
- Ejemplos:
  - `PREMIÈRS JOURS/Puzzle Tout sur moi.pdf`
  - `PREMIÈRS JOURS/infografia LA FRANCE.pdf`
  - `PREMIÈRS JOURS/LA FRANCE (1).pdf`

5. `media_audio_video` (2 archivos mp4)
- Ejemplos:
  - `U.6 DANS MON ARMOIRE/Quand le Pre Nol.wmv.mp4`
  - `U.2 MES SUPER HÉROS - LE CORPS- DESCRIPTION/jean petit qui danse karaok.mp4`

## Oportunidades directas para el proyecto

1. Plantillas de ejercicios por unidad:
- `U1 Heures`: reloj, emparejar hora-digital/analógica, dictée de horas.
- `U2 Corps`: crucigrama, completar partes del cuerpo, diálogo "j'ai mal à...".
- `U6 Vêtements`: matching prenda-descripción, colorier selon indications, oral prompts.

2. Actividades interactivas (alineadas con tu app):
- `matching` con vocabulario de ropa/cuerpo.
- `dialogue` con pistas orales tipo classroom.
- `color_match` para instrucciones visuales.
- `quiz_live` puntuable con `/api/interactive/score`.

3. Enriquecimiento multimedia:
- Usar los dos `mp4` como material de comprensión oral (preguntas por escena).
- Extraer fotogramas clave para fichas visuales.

## Propuesta técnica de siguiente paso (rápida)

1. Crear endpoint de importación de catálogo local:
- `POST /api/library/import-folder-index`
- indexa nombre, unidad, tema, tipo y ruta (sin copiar binarios al inicio).

2. Crear botón en Biblioteca:
- `Importar materiales FRANÇAIS 6º`
- genera plantillas iniciales de ejercicios desde los PDFs detectados.

3. Crear "packs por unidad":
- `Pack U1 (heures)`, `Pack U2 (corps)`, `Pack U6 (vêtements)` para generación 1-clic.
