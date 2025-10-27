PRACTICA 5: TRANSFORMADA DE HOUGH Y CUANTIFICACION VECTORIAL
================================================================

DESCRIPCION
-----------
Esta practica implementa dos tecnicas fundamentales de procesamiento digital de imagenes:

1. Transformada de Hough para deteccion de lineas
2. Cuantificacion Vectorial (algoritmo LBG) para compresion de imagenes


ESTRUCTURA DEL PROYECTO
------------------------
.
├── src/                          # Imagenes originales
│   ├── ImgHough01.png
│   ├── ImgHough03.png
│   ├── ImgHough04.png
│   ├── ImgCuantif01.jpg
│   ├── ImgCuantif02.jpg
│   └── ImgCuantif03.jpg
│
├── results/                      # Resultados procesados
│   ├── hough/                    # Resultados de Transformada de Hough
│   │   ├── ImgHough01/
│   │   ├── ImgHough03/
│   │   └── ImgHough04/
│   │
│   └── cuantificacion/           # Resultados de Cuantificacion Vectorial
│       ├── ImgCuantif01/
│       ├── ImgCuantif02/
│       └── ImgCuantif03/
│
├── hough_transform.py            # Implementacion Transformada de Hough
├── vector_quantization.py        # Implementacion Cuantificacion Vectorial
├── generate_report.py            # Generador de informe en Word
├── informe_practica_5.docx       # Informe completo (DOCUMENTO PRINCIPAL)
└── README.txt                    # Este archivo


COMO EJECUTAR
-------------

1. Asegurate de tener las dependencias instaladas:
   pip3 install opencv-python numpy matplotlib scikit-learn python-docx

2. Para ejecutar solo la Transformada de Hough:
   python3 hough_transform.py

3. Para ejecutar solo la Cuantificacion Vectorial:
   python3 vector_quantization.py

4. Para generar el informe completo en Word:
   python3 generate_report.py


RESULTADOS PRINCIPALES
-----------------------

TRANSFORMADA DE HOUGH:
Se procesaron 3 imagenes con 4 configuraciones de parametros diferentes:
- Parametros conservadores (umbral alto)
- Parametros moderados
- Parametros sensibles (umbral bajo)
- Alta resolucion angular

Los resultados muestran como los parametros afectan significativamente el numero
y calidad de las lineas detectadas.


CUANTIFICACION VECTORIAL:
Se comprimieron 3 imagenes con 2 configuraciones de bloques:

ImgCuantif01:
  - Bloques 2x2: MSE=40.34, PSNR=32.07 dB, Compresion=13.69x
  - Bloques 4x4: MSE=103.04, PSNR=28.00 dB, Compresion=53.62x

ImgCuantif02:
  - Bloques 2x2: MSE=54.48, PSNR=30.77 dB, Compresion=13.69x
  - Bloques 4x4: MSE=187.26, PSNR=25.41 dB, Compresion=53.62x

ImgCuantif03:
  - Bloques 2x2: MSE=63.57, PSNR=30.10 dB, Compresion=13.56x
  - Bloques 4x4: MSE=221.56, PSNR=24.68 dB, Compresion=46.38x


INFORME COMPLETO
----------------
Este informe incluye:
1. Introduccion teorica de ambos metodos
2. Metodologia detallada
3. Todas las imagenes procesadas en cada etapa
4. Tablas con metricas cuantitativas
5. Analisis de parametros
6. Conclusiones personales

IMAGENES GENERADAS
------------------
Total de imagenes procesadas: 42

Para cada imagen de Hough:
- Imagen original
- Escala de grises
- Bordes detectados (Canny)
- 4 versiones con diferentes parametros
- Figura comparativa completa

Para cada imagen de Cuantificacion:
- 2 configuraciones (bloques 2x2 y 4x4)
- Para cada configuracion: original, reconstruida, diferencia


CONCLUSIONES CLAVE
------------------
- La Transformada de Hough es robusta para detectar lineas pero requiere
  ajuste cuidadoso de parametros segun el tipo de imagen.

- La Cuantificacion Vectorial logra buenas tasas de compresion pero con
  un compromiso inevitable entre compresion y calidad visual.

- Bloques pequeños preservan mejor la calidad, bloques grandes comprimen mas.

- Las metricas MSE y PSNR proporcionan medidas objetivas de calidad, aunque
  no siempre coinciden perfectamente con la percepcion visual.


AUTOR
-----
Fecha: Octubre 2025
Curso: Procesamiento Digital de Imagenes
