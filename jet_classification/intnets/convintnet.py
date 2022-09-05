# Interaction net implementation in tensorflow using convolutional layers.

import numpy as np
import itertools
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as KL


class EffectsMLP(KL.Layer):
    """First MLP component of the interaction network, i.e., relational model.

    Receives the concatenated receiver, sender, and relation attributes
    (not used in this work) matrix and outputs the so-called effects matrix,
    supposed to encode the effects of the interactions between the constituents of the
    considered system.

    Related publications:
    https://arxiv.org/abs/1612.00222
    https://arxiv.org/abs/1908.05318

    THIS NETWORK IS DENOTED $f_R$ IN THESE PUBLICATIONS.
    """

    def __init__(self, neffects: int, nnodes: int, activ: str, **kwargs):

        super(EffectsMLP, self).__init__(name="relational_model", **kwargs)
        self._hid_layer_1 = KL.Conv1D(nnodes, kernel_size=1, name=f"relnet_layer_1")
        self._activ_1 = KL.Activation(activ, name=f"relnet_activ_1")
        self._hid_layer_2 = KL.Conv1D(
            int(nnodes / 2), kernel_size=1, name=f"relnet_layer_2"
        )
        self._activ_2 = KL.Activation(activ, name=f"relnet_activ_2")
        self._output_layer = KL.Conv1D(neffects, kernel_size=1, name=f"relnet_output")
        self._output_activ = KL.Activation(activ, name=f"relnet_output_activ")

    def call(self, inputs):
        x = self._hid_layer_1(inputs)
        x = self._activ_1(x)
        x = self._hid_layer_2(x)
        x = self._activ_2(x)
        x = self._output_layer(x)

        effects = self._output_activ(x)

        return effects


class DynamicsMLP(KL.Layer):
    """Second MLP component of the interaction network, i.e., the object model.

    Receives the effects matrix times the transpose of the receiver matrix and outputs
    the so-called dynamics matrix, which encodes the manifestation of the effects on
    the constituents.

    Related publications:
    https://arxiv.org/abs/1612.00222
    https://arxiv.org/abs/1908.05318

    THIS NETWORK IS DENOTED $f_O$ IN THESE PUBLICATIONS.
    """

    def __init__(self, ndynamics: int, nnodes: int, activ: str, **kwargs):

        super(DynamicsMLP, self).__init__(name="object_model", **kwargs)
        self._hid_layer_1 = KL.Conv1D(nnodes, kernel_size=1, name=f"objnet_layer_1")
        self._activ_1 = KL.Activation(activ, name=f"objnet_activ_1")
        self._hid_layer_2 = KL.Conv1D(
            int(nnodes / 2), kernel_size=1, name=f"objnet_layer_2"
        )
        self._activ_2 = KL.Activation(activ, name=f"objnet_activ_2")
        self._output_layer = KL.Conv1D(ndynamics, kernel_size=1, name=f"objnet_output")
        self._output_activ = KL.Activation(activ, name=f"objnet_output_activ")

    def call(self, inputs):
        x = self._hid_layer_1(inputs)
        x = self._activ_1(x)
        x = self._hid_layer_2(x)
        x = self._activ_2(x)
        x = self._output_layer(x)

        dynamics = self._output_activ(x)

        return dynamics


class AbstractMLP(KL.Layer):
    """Third MLP component of the interaction network, i.e., the classifier model.

    This is an optional component. It is used to distinguish the abstract quantities
    of a system. The classifier model takes the dynamics and computes abstract
    quantities about the system, e.g., for gravitational interaction it would compute
    the potential energy of the system.

    Related publications:
    https://arxiv.org/abs/1612.00222
    https://arxiv.org/abs/1908.05318

    THIS NETWORK IS DENOTED $f_A$ IN THESE PUBLICATIONS.
    """

    def __init__(self, nabs_quant: int, nnodes: int, activ: str, **kwargs):

        super(AbstractMLP, self).__init__(name="classifier_model", **kwargs)
        self._hid_layer_1 = KL.Dense(nnodes, name=f"classnet_layer_1")
        self._activ_1 = KL.Activation(activ, name=f"classnet_activ_1")
        self._output_layer = KL.Dense(nabs_quant, name=f"classnet_output")
        self._output_activ = KL.Activation("softmax", name=f"classnet_output_activ")

    def call(self, inputs):
        x = self._hid_layer_1(inputs)
        x = self._activ_1(x)
        x = self._output_layer(x)

        abstract_quantities = self._output_activ(x)

        return abstract_quantities


class ConvIntNet(keras.Model):
    """Interaction network implemented with dense layers and a classifier.

    Interaction network used to tag jets by inferring an abstract quantity, the jet
    type, from the relations between the jet constituents.

    See the following github repositories for more details:
    https://bit.ly/3PhpTcB
    https://bit.ly/39qPL55
    https://bit.ly/3FNQRUI

    For theoretical explanations, please see the papers:
    https://arxiv.org/abs/1612.00222
    https://arxiv.org/abs/1908.05318

    Attributes:
        nconst: Number of constituents the jet data has.
        nfeats: Number of features the data has.
        *_nnodes: Number of nodes that a component network has in its hidden layers.
        *_activ: The activation function after each layer for a component network.
        neffects: Number of effects, i.e., nb of output nodes for the relational net.
        ndynamics: Number of dynamical variables, i.e., output nodes for object net.
        nclasses: Number of classes, i.e., types of jets, i.e., output of the classif.
    """

    def __init__(
        self,
        nconst: int,
        nfeats: int,
        effects_nnodes: int = 30,
        dynamic_nnodes: int = 45,
        abstrac_nnodes: int = 48,
        effects_activ: str = "relu",
        dynamic_activ: str = "relu",
        abstrac_activ: str = "relu",
        neffects: int = 6,
        ndynamics: int = 6,
        nclasses: int = 5,
        summation: bool = True,
        **kwargs,
    ):

        super(ConvIntNet, self).__init__(name="conv_intnet", **kwargs)

        self.nconst = nconst
        self.nedges = nconst * (nconst - 1)
        self.nfeats = nfeats
        self.nclass = nclasses

        self._summation = summation
        self._receiver_matrix, self._sender_matrix = self._build_relation_matrices()
        self._batchnorm = KL.BatchNormalization()
        self._effects_mlp = EffectsMLP(neffects, effects_nnodes, effects_activ)
        self._dynamics_mlp = DynamicsMLP(neffects, dynamic_nnodes, dynamic_activ)
        self._abstract_mlp = AbstractMLP(nclasses, abstrac_nnodes, abstrac_activ)

    def _build_relation_matrices(self):
        """Construct the relation matrices between the graph nodes."""
        receiver_matrix = np.zeros([self.nconst, self.nedges], dtype=np.float64)
        sender_matrix = np.zeros([self.nconst, self.nedges], dtype=np.float64)
        receiver_sender_list = [
            node
            for node in itertools.product(range(self.nconst), range(self.nconst))
            if node[0] != node[1]
        ]

        for idx, (receiver, sender) in enumerate(receiver_sender_list):
            receiver_matrix[receiver, idx] = 1
            sender_matrix[sender, idx] = 1

        return receiver_matrix, sender_matrix

    def _tmul(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Do matrix multiplication of shape (I * J * K)(K * L) -> I * J * L."""
        x_shape = tf.shape(x)
        y_shape = tf.shape(y)
        return tf.reshape(
            tf.matmul(tf.reshape(x, [-1, x_shape[2]]), y), [-1, x_shape[1], y_shape[1]]
        )

    def call(self, inputs, **kwargs):

        norm_constituents = self._batchnorm(inputs)
        norm_constituents = KL.Permute((2, 1), input_shape=norm_constituents.shape[1:])(
            norm_constituents
        )

        rec_matrix = self._tmul(norm_constituents, self._receiver_matrix)
        sen_matrix = self._tmul(norm_constituents, self._sender_matrix)

        rs_matrix = KL.Concatenate(axis=1)([rec_matrix, sen_matrix])
        rs_matrix = KL.Permute((2, 1), input_shape=rs_matrix.shape[1:])(rs_matrix)
        del rec_matrix, sen_matrix

        effects = self._effects_mlp(rs_matrix)
        del rs_matrix

        effects = KL.Permute((2, 1))(effects)
        transpose_receiver_matrix = tf.transpose(self._receiver_matrix)
        effects_reduced = self._tmul(effects, transpose_receiver_matrix)
        constituents_effects_matrix = KL.Concatenate(axis=1)(
            [norm_constituents, effects_reduced]
        )
        constituents_effects_matrix = KL.Permute(
            (2, 1), input_shape=constituents_effects_matrix.shape[1:]
        )(constituents_effects_matrix)
        del effects
        del effects_reduced

        dynamics = self._dynamics_mlp(constituents_effects_matrix)

        if self._summation:
            dynamics = tf.reduce_sum(dynamics, 1)
        else:
            dynamics = KL.Flatten()(dynamics)

        abstract_quantities = self._abstract_mlp(dynamics)

        return abstract_quantities
