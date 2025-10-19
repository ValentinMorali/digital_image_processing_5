import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans

class VectorQuantizer:
    """
    Implementa la Cuantificacion Vectorial usando el algoritmo LBG (Linde-Buzo-Gray).
    Permite comprimir imagenes dividiendo en bloques y creando un diccionario de vectores codigo.
    """

    def __init__(self, image_path, block_size=(4, 4)):
        """
        Inicializa el cuantificador vectorial.

        Args:
            image_path: Ruta a la imagen a comprimir
            block_size: Tamaño de los bloques (altura, ancho)
        """
        self.image_path = image_path
        self.image_name = os.path.splitext(os.path.basename(image_path))[0]
        self.block_size = block_size
        self.original_image = cv2.imread(image_path)

        if self.original_image is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")

        # Convertir a RGB para procesamiento
        self.original_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)

        self.training_vectors = None
        self.codebook = None
        self.reconstructed_image = None
        self.indices_map = None

    def divide_into_blocks(self):
        """
        Divide la imagen en bloques no solapados del tamaño especificado.
        Crea el conjunto de entrenamiento con los vectores de cada bloque.

        Returns:
            Array de vectores de entrenamiento y dimensiones necesarias
        """
        height, width, channels = self.original_rgb.shape
        block_h, block_w = self.block_size

        # Ajustar dimensiones para que sean multiplos del tamaño de bloque
        new_height = (height // block_h) * block_h
        new_width = (width // block_w) * block_w

        # Recortar imagen si es necesario
        adjusted_image = self.original_rgb[:new_height, :new_width, :]

        # Calcular numero de bloques
        num_blocks_h = new_height // block_h
        num_blocks_w = new_width // block_w
        total_blocks = num_blocks_h * num_blocks_w

        # Dimension de cada vector (block_h * block_w * channels)
        vector_dim = block_h * block_w * channels

        # Crear array para almacenar vectores
        training_vectors = np.zeros((total_blocks, vector_dim))

        # Extraer bloques y convertir a vectores
        block_idx = 0
        for i in range(0, new_height, block_h):
            for j in range(0, new_width, block_w):
                # Extraer bloque
                block = adjusted_image[i:i+block_h, j:j+block_w, :]
                # Aplanar a vector
                vector = block.flatten()
                training_vectors[block_idx] = vector
                block_idx += 1

        self.training_vectors = training_vectors
        self.adjusted_image = adjusted_image
        self.num_blocks_h = num_blocks_h
        self.num_blocks_w = num_blocks_w

        return training_vectors, (num_blocks_h, num_blocks_w, new_height, new_width)

    def lbg_algorithm(self, codebook_size=128, epsilon=0.001, max_iterations=100):
        """
        Implementa el algoritmo LBG (Linde-Buzo-Gray) para generar el diccionario.

        El algoritmo LBG es un metodo iterativo que:
        1. Inicializa el codebook con el centroide de todos los vectores
        2. Divide cada vector codigo en dos (splitting)
        3. Asigna cada vector de entrenamiento al codigo mas cercano
        4. Actualiza los centroides
        5. Repite hasta convergencia

        Args:
            codebook_size: Numero de vectores codigo en el diccionario
            epsilon: Umbral de convergencia
            max_iterations: Numero maximo de iteraciones

        Returns:
            Codebook (diccionario de vectores codigo)
        """
        if self.training_vectors is None:
            self.divide_into_blocks()

        print(f"  Ejecutando algoritmo LBG...")
        print(f"  Vectores de entrenamiento: {len(self.training_vectors)}")
        print(f"  Tamaño del codebook: {codebook_size}")

        # Usando KMeans como implementacion eficiente del algoritmo LBG
        # KMeans es equivalente al algoritmo LBG/GLA (Generalized Lloyd Algorithm)
        kmeans = KMeans(
            n_clusters=codebook_size,
            init='k-means++',  # Inicializacion inteligente
            max_iter=max_iterations,
            tol=epsilon,
            random_state=42,
            n_init=10
        )

        # Entrenar el modelo
        kmeans.fit(self.training_vectors)

        # El codebook son los centroides
        self.codebook = kmeans.cluster_centers_

        print(f"  Codebook generado exitosamente")

        return self.codebook

    def quantize_and_reconstruct(self):
        """
        Cuantifica la imagen usando el codebook y la reconstruye.

        Este proceso:
        1. Asigna cada bloque al vector codigo mas cercano
        2. Reconstruye la imagen usando los vectores del codebook

        Returns:
            Imagen reconstruida y mapa de indices
        """
        if self.codebook is None:
            raise ValueError("Primero debes generar el codebook con lbg_algorithm()")

        print(f"  Cuantificando y reconstruyendo imagen...")

        block_h, block_w = self.block_size
        channels = 3

        # Crear imagen reconstruida
        height = self.num_blocks_h * block_h
        width = self.num_blocks_w * block_w
        reconstructed = np.zeros((height, width, channels), dtype=np.uint8)

        # Mapa de indices para analisis
        indices_map = np.zeros((self.num_blocks_h, self.num_blocks_w), dtype=np.int32)

        # Para cada vector, encontrar el codigo mas cercano
        block_idx = 0
        for i in range(self.num_blocks_h):
            for j in range(self.num_blocks_w):
                # Vector actual
                vector = self.training_vectors[block_idx]

                # Encontrar el vector codigo mas cercano (distancia euclidiana)
                distances = np.linalg.norm(self.codebook - vector, axis=1)
                closest_idx = np.argmin(distances)

                # Guardar indice
                indices_map[i, j] = closest_idx

                # Reconstruir bloque con el vector codigo
                code_vector = self.codebook[closest_idx]
                reconstructed_block = code_vector.reshape(block_h, block_w, channels)

                # Colocar en la imagen reconstruida
                y_start = i * block_h
                x_start = j * block_w
                reconstructed[y_start:y_start+block_h, x_start:x_start+block_w, :] = \
                    np.clip(reconstructed_block, 0, 255).astype(np.uint8)

                block_idx += 1

        self.reconstructed_image = reconstructed
        self.indices_map = indices_map

        print(f"  Imagen reconstruida exitosamente")

        return reconstructed, indices_map

    def calculate_metrics(self):
        """
        Calcula metricas de calidad MSE y PSNR entre imagen original y reconstruida.

        MSE (Mean Squared Error): Error cuadratico medio
        PSNR (Peak Signal-to-Noise Ratio): Relacion señal-ruido de pico

        Returns:
            Diccionario con metricas
        """
        if self.reconstructed_image is None:
            raise ValueError("Primero debes reconstruir la imagen")

        # Asegurar mismas dimensiones
        original = self.adjusted_image.astype(np.float64)
        reconstructed = self.reconstructed_image.astype(np.float64)

        # Calcular MSE
        mse = np.mean((original - reconstructed) ** 2)

        # Calcular PSNR
        if mse == 0:
            psnr = float('inf')
        else:
            max_pixel = 255.0
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

        # Calcular tasa de compresion
        original_size = original.shape[0] * original.shape[1] * original.shape[2] * 8  # bits
        # Tamaño comprimido: indices + codebook
        num_blocks = self.num_blocks_h * self.num_blocks_w
        bits_per_index = int(np.ceil(np.log2(len(self.codebook))))
        compressed_size = num_blocks * bits_per_index + len(self.codebook) * self.block_size[0] * self.block_size[1] * 3 * 8
        compression_ratio = original_size / compressed_size

        metrics = {
            'mse': mse,
            'psnr': psnr,
            'compression_ratio': compression_ratio,
            'original_size_bits': original_size,
            'compressed_size_bits': compressed_size
        }

        return metrics

    def visualize_results(self, save_path):
        """
        Crea una visualizacion comparativa de los resultados.

        Args:
            save_path: Ruta donde guardar la figura
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Imagen original
        axes[0].imshow(self.adjusted_image)
        axes[0].set_title('Imagen Original')
        axes[0].axis('off')

        # Imagen reconstruida
        axes[1].imshow(self.reconstructed_image)
        axes[1].set_title('Imagen Reconstruida')
        axes[1].axis('off')

        # Diferencia absoluta
        diff = np.abs(self.adjusted_image.astype(np.float32) -
                     self.reconstructed_image.astype(np.float32))
        diff_normalized = (diff / diff.max() * 255).astype(np.uint8)

        axes[2].imshow(diff_normalized)
        axes[2].set_title('Diferencia (Amplificada)')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def save_results(self, output_dir):
        """
        Guarda todos los resultados en el directorio especificado.

        Args:
            output_dir: Directorio de salida
        """
        os.makedirs(output_dir, exist_ok=True)

        # Guardar imagen original ajustada
        original_path = os.path.join(output_dir, 'original.png')
        cv2.imwrite(original_path, cv2.cvtColor(self.adjusted_image, cv2.COLOR_RGB2BGR))

        # Guardar imagen reconstruida
        reconstructed_path = os.path.join(output_dir, 'reconstructed.png')
        cv2.imwrite(reconstructed_path, cv2.cvtColor(self.reconstructed_image, cv2.COLOR_RGB2BGR))

        # Guardar visualizacion comparativa
        comparison_path = os.path.join(output_dir, 'comparison.png')
        self.visualize_results(comparison_path)


def process_all_quantization_images():
    """
    Procesa todas las imagenes de cuantificacion con diferentes configuraciones.
    """
    image_files = [
        'src/ImgCuantif01.jpg',
        'src/ImgCuantif02.jpg',
        'src/ImgCuantif03.jpg'
    ]

    output_base = 'results/cuantificacion'

    # Configuraciones a probar
    block_sizes = [(2, 2), (4, 4)]
    codebook_size = 128

    all_results = {}

    for img_path in image_files:
        if not os.path.exists(img_path):
            print(f"Advertencia: No se encontro {img_path}")
            continue

        print(f"\nProcesando {img_path}...")
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        all_results[img_name] = {}

        for block_size in block_sizes:
            print(f"\n  Configuracion: bloques {block_size[0]}x{block_size[1]}, codebook={codebook_size}")

            # Crear cuantificador
            vq = VectorQuantizer(img_path, block_size=block_size)

            # Dividir en bloques
            training_vectors, dims = vq.divide_into_blocks()
            print(f"  Bloques extraidos: {len(training_vectors)}")

            # Aplicar LBG
            codebook = vq.lbg_algorithm(codebook_size=codebook_size)

            # Cuantificar y reconstruir
            reconstructed, indices = vq.quantize_and_reconstruct()

            # Calcular metricas
            metrics = vq.calculate_metrics()
            print(f"  MSE: {metrics['mse']:.2f}")
            print(f"  PSNR: {metrics['psnr']:.2f} dB")
            print(f"  Compresion: {metrics['compression_ratio']:.2f}x")

            # Guardar resultados
            config_name = f"block_{block_size[0]}x{block_size[1]}_codebook_{codebook_size}"
            output_dir = os.path.join(output_base, img_name, config_name)
            vq.save_results(output_dir)

            # Almacenar resultados
            all_results[img_name][config_name] = {
                'vq': vq,
                'metrics': metrics,
                'block_size': block_size,
                'codebook_size': codebook_size,
                'output_dir': output_dir
            }

    return all_results


if __name__ == "__main__":
    print("="*70)
    print("EJERCICIO 2: COMPRESION POR CUANTIFICACION VECTORIAL (LBG)")
    print("="*70)

    results = process_all_quantization_images()

    print("\n" + "="*70)
    print("PROCESAMIENTO COMPLETADO")
    print("="*70)
    print(f"\nResultados guardados en: results/cuantificacion/")
    print(f"Total de imagenes procesadas: {len(results)}")
