import numpy as np
from si.neural_networks.layers import Layer

class DenseLayer(Layer):
    """
    Camada densa (fully connected) de uma rede neuronal.
    """
    
    def __init__(self, n_units: int, input_shape: tuple = None):
        """
        Inicializa a camada definindo o número de neurónios e o formato de entrada.
        """

        super().__init__()
        self.n_units = n_units
        self._input_shape = input_shape
        
        # Parâmetros a estimaar
        self.input = None
        self.output = None
        self.weights = None
        self.biases = None
        self.optimizer = None

    def initialize(self, optimizer=None):
        """
        Inicializa os pesos com valores aleatórios entre -0.5 e 0.5, e os biases a zero.
        Inicializa também o otimizador, se fornecido.
        """
        # Extrair o número de features
        n_features = self._input_shape[0]
        
        self.weights = np.random.uniform(-0.5, 0.5, (n_features, self.n_units))
        
        self.biases = np.zeros((1, self.n_units))
        
        self.optimizer = optimizer
        
        return self

    def parameters(self):
        """
        Devolve o número total de parâmetros da camada (pesos + biases).
        """
        return np.prod(self.weights.shape) + np.prod(self.biases.shape)

    def forward_propagation(self, input: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Realiza a propagação para a frente (forward propagation).
        """
        self.input = input
        
        self.output = np.dot(self.input, self.weights) + self.biases
        
        return self.output
    
    def backward_propagation(self, output_error: np.ndarray):
        """
        Realiza a backward propagation.
        """
        raise NotImplementedError("A retropropagação será implementada de seguida.")

    def output_shape(self) -> tuple:
        """
        Devolve a forma (shape) do output da camada.
        """
        return (self.n_units,)


if __name__ == '__main__':
    layer = DenseLayer(4, (3,)).initialize()
    out = layer.forward_propagation(np.random.rand(10, 3))
    
    assert out.shape == (10, 4)
    assert layer.parameters() == 3 * 4 + 4  # 16
    assert layer.output_shape() == (4,)
    
    print(f"Formato do output: {out.shape}")
    print(f"Número de parâmetros: {layer.parameters()}")