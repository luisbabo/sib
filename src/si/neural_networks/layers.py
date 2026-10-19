from abc import ABCMeta, abstractmethod
import numpy as np

class Layer(metaclass=ABCMeta):
    """
    Classe base abstrata para todas as camadas de uma rede neuronal.
    """

    @abstractmethod
    def forward_propagation(self, input, training):
        """
        Calcula o output de uma camada dado o input.
        """
        raise NotImplementedError

    @abstractmethod
    def backward_propagation(self, output_error):
        """
        Faz a retropropagação (backpropagation) do erro do output.
        """
        raise NotImplementedError

    @abstractmethod
    def output_shape(self):
        """
        Devolve a forma (shape) do output da camada.
        """
        raise NotImplementedError

    @abstractmethod
    def parameters(self):
        """
        Devolve o número de parâmetros treináveis da camada.
        """
        raise NotImplementedError

    def set_input_shape(self, input_shape):
        """
        Define a forma (shape) do input da camada.
        """
        self._input_shape = input_shape

    def input_shape(self):
        """
        Devolve a forma (shape) do input da camada.
        """
        return self._input_shape

    def layer_name(self):
        """
        Devolve o nome da classe da camada.
        """
        return self.__class__.__name__