import tensorflow as tf

class CaffeLSTMCell(tf.contrib.rnn.RNNCell):

  def __init__(self, num_units,
               initializer=None,
               activation=tf.nn.tanh):
    """Initialize the parameters for an LSTM cell.
    Args:
      num_units: int, The number of units in the LSTM cell
      initializer: (optional) The initializer to use for the weight and
        projection matrices.
      activation: Activation function of the inner states.
    """

    self._num_units = num_units
    self._initializer = initializer
    self._activation = activation

    self._state_size = tf.contrib.rnn.LSTMStateTuple(num_units, num_units)
    self._output_size = num_units

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  def __call__(self, inputs, state, scope=None):
    """Run one step of LSTM.
    Args:
      inputs: input Tensor, 2D, batch x num_units.
      state: This must be a tuple of state Tensors,
      both `2-D`, with column sizes `c_state` and `m_state`.
      scope: VariableScope for the created subgraph; defaults to "lstm_cell".
    Returns:
      A tuple containing:
      - A `2-D, [batch x output_dim]`, Tensor representing the output of the
        LSTM after reading `inputs` when previous state was `state`.
        Here output_dim is:
           num_proj if num_proj was set,
           num_units otherwise.
      - Tensor(s) representing the new state of LSTM after reading `inputs` when
        the previous state was `state`.  Same type and shape(s) as `state`.
    Raises:
      ValueError: If input size cannot be inferred from inputs via
        static shape inference.
    """

    with tf.variable_scope('LSTM'):

      (cell_state_prev, cell_outputs_prev) = state
      dtype = inputs.dtype

      lstm_concat = tf.concat([inputs, cell_outputs_prev], axis=1)
      inputs_shape = lstm_concat.get_shape().as_list()[1]

      peephole_concat = tf.concat([lstm_concat, cell_state_prev], axis=1)
      peephole_shape = peephole_concat.get_shape().as_list()[1]

      with tf.variable_scope('block_input'):
        weights = tf.get_variable('weights',
                shape=[inputs_shape, self._num_units],
                dtype=dtype, initializer=self._initializer)
        biases = tf.get_variable('biases', shape=[self._num_units], dtype=dtype,
                initializer=tf.zeros_initializer())
        block_input = self._activation(tf.matmul(lstm_concat, weights) + biases)

      with tf.variable_scope('input_gate'):
        weights = tf.get_variable('weights',
                shape=[peephole_shape, self._num_units],
                dtype=dtype, initializer=self._initializer)
        biases = tf.get_variable('biases', shape=[self._num_units], dtype=dtype,
                initializer=tf.zeros_initializer())
        input_gate = tf.nn.sigmoid(tf.matmul(peephole_concat, weights) + biases)

        input_mult = input_gate * block_input

      with tf.variable_scope('forget_gate'):
        weights = tf.get_variable('weights',
                shape=[peephole_shape, self._num_units],
                dtype=dtype, initializer=self._initializer)
        biases = tf.get_variable('biases', shape=[self._num_units], dtype=dtype,
                initializer=tf.ones_initializer())
        forget_gate = tf.nn.sigmoid(tf.matmul(peephole_concat, weights) + biases)

        forget_mult = forget_gate * cell_state_prev

        cell_state_new = input_mult + forget_mult
        cell_state_activated = self._activation(cell_state_new)

      with tf.variable_scope('output_gate'):
        output_concat = tf.concat([lstm_concat, cell_state_new], axis=1)
        output_concat_shape = output_concat.get_shape().as_list()[1]
        weights = tf.get_variable('weights',
                shape=[output_concat_shape, self._num_units],
                dtype=dtype, initializer=self._initializer)
        biases = tf.get_variable('biases', shape=[self._num_units], dtype=dtype,
                initializer=tf.zeros_initializer())
        output_gate = tf.nn.sigmoid(tf.matmul(output_concat, weights) + biases)

        cell_outputs_new = output_gate * cell_state_activated

    new_state = tf.contrib.rnn.LSTMStateTuple(cell_state_new, cell_outputs_new)
    return cell_outputs_new, new_state

