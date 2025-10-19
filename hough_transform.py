import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class HoughLineDetector:
    """
    Clase para detectar lineas usando la Transformada de Hough.
    Implementa deteccion de bordes y transformada de Hough con diferentes parametros.
    """

    def __init__(self, image_path):
        """
        Inicializa el detector con la ruta de la imagen.

        Args:
            image_path: Ruta a la imagen a procesar
        """
        self.image_path = image_path
        self.image_name = os.path.splitext(os.path.basename(image_path))[0]
        self.original_image = cv2.imread(image_path)

        if self.original_image is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")

        self.gray_image = None
        self.edges = None
        self.results = {}

    def convert_to_grayscale(self):
        """
        Convierte la imagen original a escala de grises.
        """
        self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        return self.gray_image

    def detect_edges(self, low_threshold=50, high_threshold=150, save_path=None):
        """
        Aplica el detector de bordes Canny.

        Args:
            low_threshold: Umbral inferior para Canny
            high_threshold: Umbral superior para Canny
            save_path: Ruta donde guardar la imagen de bordes
        """
        if self.gray_image is None:
            self.convert_to_grayscale()

        # Aplicar un ligero desenfoque para reducir ruido
        blurred = cv2.GaussianBlur(self.gray_image, (5, 5), 1.5)

        # Detectar bordes con Canny
        self.edges = cv2.Canny(blurred, low_threshold, high_threshold)

        if save_path:
            cv2.imwrite(save_path, self.edges)

        return self.edges

    def detect_lines_hough(self, rho=1, theta=np.pi/180, threshold=100,
                          min_line_length=50, max_line_gap=10):
        """
        Detecta lineas usando la Transformada de Hough Probabilistica.

        Args:
            rho: Resolucion de distancia en pixeles (delta_rho)
            theta: Resolucion angular en radianes (delta_theta)
            threshold: Umbral minimo de votos en el acumulador
            min_line_length: Longitud minima de linea
            max_line_gap: Distancia maxima entre segmentos de linea

        Returns:
            Array de lineas detectadas
        """
        if self.edges is None:
            raise ValueError("Primero debes detectar bordes con detect_edges()")

        # Transformada de Hough probabilistica
        lines = cv2.HoughLinesP(
            self.edges,
            rho=rho,
            theta=theta,
            threshold=threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap
        )

        return lines

    def detect_lines_standard_hough(self, rho=1, theta=np.pi/180, threshold=100):
        """
        Detecta lineas usando la Transformada de Hough estandar.

        Args:
            rho: Resolucion de distancia en pixeles
            theta: Resolucion angular en radianes
            threshold: Umbral minimo de votos

        Returns:
            Array de lineas en formato (rho, theta)
        """
        if self.edges is None:
            raise ValueError("Primero debes detectar bordes")

        lines = cv2.HoughLines(self.edges, rho, theta, threshold)
        return lines

    def draw_lines_on_image(self, lines, color=(0, 255, 0), thickness=2):
        """
        Dibuja las lineas detectadas sobre la imagen original.

        Args:
            lines: Array de lineas detectadas
            color: Color de las lineas en formato BGR
            thickness: Grosor de las lineas

        Returns:
            Imagen con lineas superpuestas
        """
        result_image = self.original_image.copy()

        if lines is None:
            return result_image

        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result_image, (x1, y1), (x2, y2), color, thickness)

        return result_image

    def draw_standard_hough_lines(self, lines, color=(0, 255, 0), thickness=2):
        """
        Dibuja lineas de la Transformada de Hough estandar.

        Args:
            lines: Lineas en formato (rho, theta)
            color: Color de las lineas
            thickness: Grosor
        """
        result_image = self.original_image.copy()

        if lines is None:
            return result_image

        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            # Calcular puntos extremos de la linea
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(result_image, (x1, y1), (x2, y2), color, thickness)

        return result_image

    def process_with_multiple_parameters(self, param_sets, output_dir):
        """
        Procesa la imagen con multiples conjuntos de parametros para analizar su efecto.

        Args:
            param_sets: Lista de diccionarios con parametros
            output_dir: Directorio donde guardar resultados
        """
        results = []

        for i, params in enumerate(param_sets):
            # Detectar lineas con estos parametros
            lines = self.detect_lines_hough(**params)

            # Dibujar lineas
            result_img = self.draw_lines_on_image(lines)

            # Guardar resultado
            save_path = os.path.join(output_dir,
                                    f"{self.image_name}_params_{i+1}.png")
            cv2.imwrite(save_path, result_img)

            num_lines = len(lines) if lines is not None else 0
            results.append({
                'params': params,
                'num_lines': num_lines,
                'image_path': save_path
            })

        return results

    def create_comparison_figure(self, lines_dict, save_path):
        """
        Crea una figura con comparacion de resultados con diferentes parametros.

        Args:
            lines_dict: Diccionario con {nombre: lineas}
            save_path: Ruta donde guardar la figura
        """
        num_results = len(lines_dict)
        fig, axes = plt.subplots(1, num_results + 2, figsize=(5*(num_results+2), 5))

        # Imagen original
        axes[0].imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Imagen Original')
        axes[0].axis('off')

        # Bordes detectados
        axes[1].imshow(self.edges, cmap='gray')
        axes[1].set_title('Bordes (Canny)')
        axes[1].axis('off')

        # Resultados con diferentes parametros
        for idx, (name, lines) in enumerate(lines_dict.items()):
            result_img = self.draw_lines_on_image(lines)
            axes[idx + 2].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            num_lines = len(lines) if lines is not None else 0
            axes[idx + 2].set_title(f'{name}\n({num_lines} lineas)')
            axes[idx + 2].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def process_all_hough_images():
    """
    Procesa todas las imagenes de Hough con diferentes configuraciones de parametros.
    Guarda resultados intermedios y finales en la carpeta results/hough/
    """

    # Imagenes a procesar
    image_files = [
        'src/ImgHough01.png',
        'src/ImgHough03.png',
        'src/ImgHough04.png'
    ]

    output_base = 'results/hough'

    # Diferentes conjuntos de parametros para experimentar
    # Estos parametros se ajustaran segun el tipo de imagen

    all_results = {}

    for img_path in image_files:
        if not os.path.exists(img_path):
            print(f"Advertencia: No se encontro {img_path}")
            continue

        print(f"\nProcesando {img_path}...")

        # Crear detector
        detector = HoughLineDetector(img_path)
        img_name = detector.image_name

        # Crear directorio para esta imagen
        img_output_dir = os.path.join(output_base, img_name)
        os.makedirs(img_output_dir, exist_ok=True)

        # Guardar imagen original
        original_path = os.path.join(img_output_dir, f'{img_name}_original.png')
        cv2.imwrite(original_path, detector.original_image)

        # Convertir a escala de grises
        gray = detector.convert_to_grayscale()
        gray_path = os.path.join(img_output_dir, f'{img_name}_gray.png')
        cv2.imwrite(gray_path, gray)

        # Detectar bordes
        edges_path = os.path.join(img_output_dir, f'{img_name}_edges.png')
        detector.detect_edges(save_path=edges_path)

        # Detectar lineas con diferentes parametros para analizar su efecto
        param_configs = {
            'Parametros Conservadores (umbral alto)': {
                'rho': 1,
                'theta': np.pi/180,
                'threshold': 150,
                'min_line_length': 100,
                'max_line_gap': 10
            },
            'Parametros Moderados': {
                'rho': 1,
                'theta': np.pi/180,
                'threshold': 80,
                'min_line_length': 50,
                'max_line_gap': 15
            },
            'Parametros Sensibles (umbral bajo)': {
                'rho': 1,
                'theta': np.pi/180,
                'threshold': 40,
                'min_line_length': 30,
                'max_line_gap': 20
            },
            'Alta Resolucion Angular': {
                'rho': 1,
                'theta': np.pi/360,  # Mayor precision angular
                'threshold': 100,
                'min_line_length': 50,
                'max_line_gap': 10
            }
        }

        lines_results = {}
        for config_name, params in param_configs.items():
            lines = detector.detect_lines_hough(**params)
            lines_results[config_name] = lines

            # Guardar imagen con lineas
            img_with_lines = detector.draw_lines_on_image(lines)
            safe_name = config_name.replace(' ', '_').replace('(', '').replace(')', '')
            save_path = os.path.join(img_output_dir, f'{img_name}_{safe_name}.png')
            cv2.imwrite(save_path, img_with_lines)

            num_lines = len(lines) if lines is not None else 0
            print(f"  {config_name}: {num_lines} lineas detectadas")

        # Crear figura comparativa
        comparison_path = os.path.join(img_output_dir, f'{img_name}_comparison.png')
        detector.create_comparison_figure(lines_results, comparison_path)

        # Guardar resultados de esta imagen
        all_results[img_name] = {
            'detector': detector,
            'lines_results': lines_results,
            'param_configs': param_configs,
            'output_dir': img_output_dir
        }

    return all_results


if __name__ == "__main__":
    print("="*60)
    print("EJERCICIO 1: DETECCION DE LINEAS CON TRANSFORMADA DE HOUGH")
    print("="*60)

    results = process_all_hough_images()

    print("\n" + "="*60)
    print("PROCESAMIENTO COMPLETADO")
    print("="*60)
    print(f"\nResultados guardados en: results/hough/")
    print(f"Total de imagenes procesadas: {len(results)}")
