"""
Script para generar un resumen visual de todos los resultados obtenidos
"""

def print_resumen():
    print("\n" + "="*80)
    print("RESUMEN DE RESULTADOS - PRACTICA 5")
    print("="*80)

    print("\n1. TRANSFORMADA DE HOUGH - DETECCION DE LINEAS")
    print("-" * 80)

    hough_results = {
        'ImgHough01': {
            'Conservadores': 0,
            'Moderados': 21,
            'Sensibles': 161,
            'Alta Resolucion': 13
        },
        'ImgHough03': {
            'Conservadores': 2,
            'Moderados': 2,
            'Sensibles': 7,
            'Alta Resolucion': 2
        },
        'ImgHough04': {
            'Conservadores': 7,
            'Moderados': 38,
            'Sensibles': 117,
            'Alta Resolucion': 24
        }
    }

    for imagen, resultados in hough_results.items():
        print(f"\n{imagen}:")
        for config, num_lineas in resultados.items():
            print(f"  {config:20s}: {num_lineas:4d} lineas detectadas")

    print("\n\nOBSERVACIONES:")
    print("  - Los parametros conservadores detectan solo lineas muy prominentes")
    print("  - Los parametros sensibles capturan muchas mas lineas pero con mas ruido")
    print("  - La resolucion angular afecta la precision de la orientacion detectada")

    print("\n" + "="*80)
    print("2. CUANTIFICACION VECTORIAL - COMPRESION DE IMAGENES")
    print("-" * 80)

    cuantif_results = {
        'ImgCuantif01': {
            '2x2': {'mse': 40.34, 'psnr': 32.07, 'comp': 13.69},
            '4x4': {'mse': 103.04, 'psnr': 28.00, 'comp': 53.62}
        },
        'ImgCuantif02': {
            '2x2': {'mse': 54.48, 'psnr': 30.77, 'comp': 13.69},
            '4x4': {'mse': 187.26, 'psnr': 25.41, 'comp': 53.62}
        },
        'ImgCuantif03': {
            '2x2': {'mse': 63.57, 'psnr': 30.10, 'comp': 13.56},
            '4x4': {'mse': 221.56, 'psnr': 24.68, 'comp': 46.38}
        }
    }

    print("\nTabla de Metricas (Codebook: 128 vectores):")
    print("-" * 80)
    print(f"{'Imagen':<20} {'Bloque':<8} {'MSE':>10} {'PSNR (dB)':>12} {'Comp. (x)':>12}")
    print("-" * 80)

    for imagen, configs in cuantif_results.items():
        for bloque, metricas in configs.items():
            print(f"{imagen:<20} {bloque:<8} {metricas['mse']:>10.2f} "
                  f"{metricas['psnr']:>12.2f} {metricas['comp']:>12.2f}")

    print("\n\nOBSERVACIONES:")
    print("  - Bloques 2x2: Mejor calidad (PSNR > 30 dB), menor compresion (~14x)")
    print("  - Bloques 4x4: Mayor compresion (~46-54x), calidad reducida (PSNR 24-28 dB)")
    print("  - Imagenes con mas textura tienen mayor MSE y menor PSNR")
    print("  - El algoritmo LBG converge eficientemente en pocas iteraciones")

    print("\n" + "="*80)
    print("3. ARCHIVOS GENERADOS")
    print("-" * 80)

    print("\nCodigo fuente:")
    print("  - hough_transform.py           : Implementacion Transformada de Hough")
    print("  - vector_quantization.py       : Implementacion algoritmo LBG")
    print("  - generate_report.py           : Generador de informe Word")

    print("\nResultados:")
    print("  - results/hough/               : 3 imagenes procesadas con Hough")
    print("  - results/cuantificacion/      : 3 imagenes comprimidas (6 configs)")
    print("  - informe_practica_5.docx      : INFORME COMPLETO (8.2 MB)")

    print("\nTotal de imagenes generadas: 42")

    print("\n" + "="*80)
    print("4. ANALISIS COMPARATIVO")
    print("-" * 80)

    print("\nTRANSFORMADA DE HOUGH:")
    print("  Ventajas:")
    print("    + Robusta ante ruido y discontinuidades")
    print("    + Detecta lineas parcialmente ocultas")
    print("    + Parametrizable para diferentes necesidades")
    print("  \n  Desafios:")
    print("    - Requiere ajuste de parametros segun la imagen")
    print("    - Costosa computacionalmente para imagenes grandes")
    print("    - Puede generar falsos positivos con parametros inadecuados")

    print("\nCUANTIFICACION VECTORIAL:")
    print("  Ventajas:")
    print("    + Aprovecha correlacion espacial entre pixeles")
    print("    + Tasas de compresion ajustables segun necesidad")
    print("    + Calidad predecible mediante metricas objetivas")
    print("  \n  Desafios:")
    print("    - Compromiso inevitable entre compresion y calidad")
    print("    - Entrenamiento del codebook es costoso")
    print("    - Efecto de blocado visible con bloques grandes")

    print("\n" + "="*80)
    print("DOCUMENTO PRINCIPAL: informe_practica_5.docx")
    print("="*80)
    print("\nEl informe completo contiene:")
    print("  1. Introduccion teorica detallada")
    print("  2. Metodologia completa de implementacion")
    print("  3. Todas las imagenes procesadas en cada etapa")
    print("  4. Tablas y graficos con resultados cuantitativos")
    print("  5. Analisis de efectos de parametros")
    print("  6. Conclusiones personales y reflexiones")
    print("\nEscrito con lenguaje natural de estudiante de ingenieria,")
    print("sin uso de emojis, explicando conceptos de forma clara y directa.")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    print_resumen()
