# Interaction net implementation in tensorflow using quantised convolutional layers.

import numpy as np
import itertools

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as KL
import qkeras


class EffectsMLP(KL.Layer):
    """The first MLP of the interaction network, that receives the concatenated
    receiver, sender, and relation attributes (not used in this work) matrix and
    outputs the so-called effects matrix, supposed to encode the effects of the
    interactions between the constituents of the considered system.

    In the publications:
    https://arxiv.org/abs/1612.00222
    https://arxiv.org/abs/1908.05318
    this network is denoted f_r.
    """

    def __init__(self, neffects: int, nnodes: int, activ: str, nbits: int, **kwargs):

        super(EffectsMLP, self).__init__(name="fr", **kwargs)
        self._input_layer = qkeras.QConv1D(
            nnodes,
            kernel_size=1,
            kernel_quantizer=nbits,
            bias_quantizer=nbits,
            name=f"eff_layer_1",
        )
        self._activ_1 = qkeras.QActivation(activ, name=f"eff_activ_1")
        self._hidden_layer = qkeras.QConv1D(
            int(nnodes / 2),
            kernel_size=1,
            kernel_quantizer=nbits,
            bias_quantizer=nbits,
            name=f"eff_layer_2",
        )
        self._activ_2 = qkeras.QActivation(activ, name=f"eff_activ_2")
        self._output_layer = qkeras.QConv1D(
            neffects,
            kernel_size=1,
            kernel_quantizer=nbits,
            bias_quantizer=nbits,
            name=f"eff_layer_3",
        )
        self._activ_3 = qkeras.QActivation(activ, name=f"eff_activ_3")

    def call(self, inputs):
        proc_data = self._input_layer(inputs)
        proc_data = self._activ_1(proc_data)
        proc_data = self._hidden_layer(proc_data)
        proc_data = self._activ_2(proc_data)
        proc_data = self._output_layer(proc_data)
        effects = self._activ_3(proc_data)

        return effects


class DynamicsMLP(KL.Layer):
    """The second MLP of the interaction network, that receives the effects matrix
    times the transpose of the receiver matrix and outputs the so-called dynamics
    matrix, which encodes the manifestation of the effects on the constituents.

    In the publications:
    https://arxiv.org/abs/1612.00222
    https://arxiv.org/abs/1908.05318
    this network is denoted f_o.
    """

    def __init__(self, ndynamics: int, nnodes: int, activ: str, nbits: int, **kwargs):

        super(DynamicsMLP, self).__init__(name="fo", **kwargs)
        self._input_layer = qkeras.QConv1D(
            nnodes,
            kernel_size=1,
            kernel_quantizer=nbits,
            bias_quantizer=nbits,
            name=f"dyn_layer_1",
        )
        self._activ_1 = qkeras.QActivation(activ, name=f"dyn_activ_1")
        self._hidden_layer = qkeras.QConv1D(
            int(nnodes / 2),
            kernel_size=1,
            kernel_quantizer=nbits,
            bias_quantizer=nbits,
            name=f"dyn_layer_2",
        )
        self._activ_2 = qkeras.QActivation(activ, name=f"dyn_activ_2")
        self._output_layer = qkeras.QConv1D(
            ndynamics,
            kernel_size=1,
            kernel_quantizer=nbits,
            bias_quantizer=nbits,
            name=f"dyn_layer_3",
        )
        self._activ_3 = qkeras.QActivation(activ, name=f"dyn_activ_3")

    def call(self, inputs):
        proc_data = self._input_layer(inputs)
        proc_data = self._activ_1(proc_data)
        proc_data = self._hidden_layer(proc_data)
        proc_data = self._activ_2(proc_data)
        proc_data = self._output_layer(proc_data)
        dynamics = self._activ_3(proc_data)

        return dynamics


class AbstractMLP(KL.Layer):
    """Final and optional MLP of the interaction network. This MLP takes the dynamics
    and computes abstract quantities about the system, e.g., for gravitational
    interaction it would compute the potential energy of the system.

    In the publications:
    https://arxiv.org/abs/1612.00222
    https://arxiv.org/abs/1908.05318
    this network is not denoted in a specific way, but some people use f_c.
    """

    def __init__(self, nabs_quant: int, nnodes: int, activ: str, nbits: int, **kwargs):

        super(AbstractMLP, self).__init__(name="fc", **kwargs)
        self._input_layer = qkeras.QDense(
            nnodes, kernel_quantizer=nbits, bias_quantizer=nbits, name=f"abs_layer_1"
        )
        self._activ_1 = qkeras.QActivation(activ, name=f"abs_activ_1")
        self._output_layer = qkeras.QDense(
            nabs_quant,
            kernel_quantizer=nbits,
            bias_quantizer=nbits,
            name=f"abs_layer_2",
        )
        self._activ_2 = KL.Activation("softmax", name=f"abs_activ_2")

    def call(self, inputs):
        proc_data = self._input_layer(inputs)
        proc_data = self._activ_1(proc_data)
        proc_data = self._output_layer(proc_data)
        abstract_quantities = self._activ_2(proc_data)

        return abstract_quantities


class QConvIntNet(keras.Model):
    """Quantized interaction network implemented with convolutional layers. Use it to
    tag jets by inferring abstract quantities from the relations between the jet
    constituents.

    See the following githubrepositories for more details:
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
        nbits: The number of bits to quantise the floats to.
    """

    def __init__(
        self,
        nconst: int,
        nfeats: int,
        nclasses: int = 5,
        effects_nnodes: int = 30,
        dynamic_nnodes: int = 45,
        abstrac_nnodes: int = 48,
        effects_activ: str = "relu",
        dynamic_activ: str = "relu",
        abstrac_activ: str = "relu",
        neffects: int = 6,
        ndynamics: int = 6,
        nbits: int = 8,
        summation: bool = True,
        **kwargs,
    ):
        super(QConvIntNet, self).__init__(name="quantized_intnet", **kwargs)

        self.nconst = nconst
        self.nedges = nconst * (nconst - 1)
        self.nfeats = nfeats
        self.nclass = nclasses
        self.nbits = nbits

        self._summation = summation

        effects_activ = self._format_qactivation(effects_activ, nbits)
        dynamic_activ = self._format_qactivation(dynamic_activ, nbits)
        abstrac_activ = self._format_qactivation(abstrac_activ, nbits)
        nbits = self._format_quantiser(nbits)

        self._batchnorm = KL.BatchNormalization()
        self._receiver_matrix, self._sender_matrix = self._build_relation_matrices()
        self._receiver_matrix_multiplication = qkeras.QConv1D(
            self._receiver_matrix.shape[1],
            kernel_size=1,
            kernel_quantizer=nbits,
            bias_quantizer=nbits,
            use_bias=False,
            trainable=False,
            name="ORr",
        )
        self._sender_matrix_multiplication = qkeras.QConv1D(
            self._sender_matrix.shape[1],
            kernel_size=1,
            kernel_quantizer=nbits,
            bias_quantizer=nbits,
            use_bias=False,
            trainable=False,
            name="ORs",
        )
        self._effects_mlp = EffectsMLP(neffects, effects_nnodes, effects_activ, nbits)
        self._effects_matrix_reduction = qkeras.QConv1D(
            np.transpose(self._receiver_matrix).shape[1],
            kernel_size=1,
            kernel_quantizer=nbits,
            bias_quantizer=nbits,
            use_bias=False,
            trainable=False,
            name="Ebar",
        )
        self._dynamics_mlp = DynamicsMLP(neffects, dynamic_nnodes, dynamic_activ, nbits)
        self._abstract_mlp = AbstractMLP(nclasses, abstrac_nnodes, abstrac_activ, nbits)

    def _format_quantiser(self, nbits: int):
        """Format the quantisation of the ml floats in a QKeras way."""
        if nbits == 1:
            self.nbits = "binary(alpha=1)"
        elif nbits == 2:
            self.nbits = "ternary(alpha=1)"
        else:
            self.nbits = f"quantized_bits({nbits}, 0, alpha=1)"

    def _format_qactivation(self, activation: str, nbits: int) -> str:
        """Format the activation function strings in a QKeras friendly way."""
        return f"quantized_{activation}({nbits}, 0)"

    def _build_relation_matrices(self):
        """Construct the relation matrices between the graph nodes."""
        receiver_matrix = np.zeros([self.nconst, self.nedges], dtype=np.float32)
        sender_matrix = np.zeros([self.nconst, self.nedges], dtype=np.float32)
        receiver_sender_list = [
            node
            for node in itertools.product(range(self.nconst), range(self.nconst))
            if node[0] != node[1]
        ]

        for idx, (receiver, sender) in enumerate(receiver_sender_list):
            receiver_matrix[receiver, idx] = 1
            sender_matrix[sender, idx] = 1

        return receiver_matrix, sender_matrix

    def call(self, inputs, **kwargs):
        norm_constituents = self._batchnorm(inputs)
        norm_constituents = KL.Permute((2, 1), input_shape=norm_constituents.shape[1:])(
            norm_constituents
        )

        rec_matrix = self._receiver_matrix_multiplication(norm_constituents)
        sen_matrix = self._sender_matrix_multiplication(norm_constituents)

        rs_matrix = KL.Concatenate(axis=1)([rec_matrix, sen_matrix])
        rs_matrix = KL.Permute((2, 1), input_shape=rs_matrix.shape[1:])(rs_matrix)
        del rec_matrix, sen_matrix

        effects = self._effects_mlp(rs_matrix)
        del rs_matrix

        effects = KL.Permute((2, 1))(effects)
        effects_reduced = self._effects_matrix_reduction(effects)
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
            print("Summation layer on!")
            dynamics = tf.reduce_sum(dynamics, 1)
        else:
            dynamics = KL.Flatten()(dynamics)

        abstract_quantities = self._abstract_mlp(dynamics)

        return abstract_quantities
