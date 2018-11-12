---
layout: post
title:  "Attention mechanism in NLP"
date:   2018-11-12
desc: ""
keywords: "Attention，Seq2Seq"
categories: [NLP]
tags: [NLP]
icon: icon-NLP
---

Attention现在基本在所有NLP任务中都可以，感觉不用就不能发论文了。。。

今天看别人的代码发现attention也有好几种，从公式、代码角度记录下，然后分析分析物理意义

# 1. 隐含变量attention
在论文[Hierarchical Attention Networks for Document Classification](http://www.aclweb.org/anthology/N16-1174)
中使用到了attention机制，模型结构图如下：

![HAN model](/images/HAN.png 'HAN model')


使用的attention计算如下：

$$ u_i =  tanh(W_sh_i + b_s) $$

$$ \alpha_i = \frac{exp(u_i^T u_s)}{\sum_i exp(u_i^T u_s)} $$

$$ v = \sum_i \alpha_i h_i  $$

物理含义就是直接使用一个变量$u_s$作为attention，与每个$h_i$点乘算权重，总共的变量只有三个：$W_s$，$b_s$, $u_s$。
这种方法得到的attention向量是完全没法解释的。一种实现的代码如下：
```
def attention(atten_inputs, atten_size):
    ## attention mechanism uses Ilya Ivanov's implementation(https://github.com/ilivans/tf-rnn-attention)
    print('attention inputs: '+str(atten_inputs))
    max_time = int(atten_inputs.shape[1])
    print("max time length: "+str(max_time))
    combined_hidden_size = int(atten_inputs.shape[2])
    print("combined hidden size: "+str(combined_hidden_size))
    W_omega = tf.Variable(tf.random_normal([combined_hidden_size, atten_size], stddev=0.1, dtype=tf.float32))
    b_omega = tf.Variable(tf.random_normal([atten_size], stddev=0.1, dtype=tf.float32))
    u_omega = tf.Variable(tf.random_normal([atten_size], stddev=0.1, dtype=tf.float32))

    v = tf.tanh(tf.matmul(tf.reshape(atten_inputs, [-1, combined_hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
    print("v: "+str(v))
    # u_omega is the summarizing question vector
    vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
    print("vu: "+str(vu))
    exps = tf.reshape(tf.exp(vu), [-1, max_time])
    print("exps: "+str(exps))
    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])
    print("alphas: "+str(alphas))
    atten_outs = tf.reduce_sum(atten_inputs * tf.reshape(alphas, [-1, max_time, 1]), 1)
    print("atten outs: "+str(atten_outs))
    return atten_outs, alphas
```
上面的代码与公式的对应关系如下：

|   代码    |    公式    |
|:--------:|:----------:|
|   v      |    $u_i$   |
|alphas    | $\alpha_i$ |
|atten_outs|   $v$      |

# 2. Seq2Seq中的attention
seq2seq中用到的attention，是一种query，output的模式。在decoder阶段，将每个$s_{t-1}$
与encoder中的$h_i$做一个match，得到一个标量$e_{t-1,i}$，再经过softmax得到attention
$\alpha$。这里的match一般有三种：
* 直接的向量相似度（例如cosine similarity，点乘）
* 用一个MLP（[TensorFlow BahdanauAttention](https://arxiv.org/pdf/1409.0473.pdf)）
* 矩阵变换，或者叫bilinear similarity（[TensorFlow LuongAttention](https://arxiv.org/pdf/1508.04025.pdf)）
以上三种在[Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/pdf/1508.04025.pdf)
中分别对应为：dot，concat，general

现在常用的是TensorFlow中的BahdanauAttention，公式为：

$$e_{ij}=a(s_{i-1},h_j)$$

$$\alpha = \frac{exp(e_{ij})}{\sum_{k=1}^{T_x} exp(e_{ik})}$$

$$c_i = \sum_{j=1}^{T_x} \alpha_{ij}h_j$$

TensorFlow(bahdanau)实现的代码：
```
def __call__(self, query, previous_alignments):
  """Score the query based on the keys and values.

  Args:
    query: Tensor of dtype matching `self.values` and shape
      `[batch_size, query_depth]`.
    previous_alignments: Tensor of dtype matching `self.values` and shape
      `[batch_size, alignments_size]`
      (`alignments_size` is memory's `max_time`).

  Returns:
    alignments: Tensor of dtype matching `self.values` and shape
      `[batch_size, alignments_size]` (`alignments_size` is memory's
      `max_time`).
  """
  with variable_scope.variable_scope(None, "bahdanau_attention", [query]):
    processed_query = self.query_layer(query) if self.query_layer else query  # 将query转化成attention size
    score = _bahdanau_score(processed_query, self._keys, self._normalize)  # 计算原始attention，即$e_{ij}$
  alignments = self._probability_fn(score, previous_alignments)  # 默认是一个softmax操作，返回的是$\alpha$
  return alignments


def _bahdanau_score(processed_query, keys, normalize):
  """Implements Bahdanau-style (additive) scoring function.

  This attention has two forms.  The first is Bhandanau attention,
  as described in:

  Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
  "Neural Machine Translation by Jointly Learning to Align and Translate."
  ICLR 2015. https://arxiv.org/abs/1409.0473

  The second is the normalized form.  This form is inspired by the
  weight normalization article:

  Tim Salimans, Diederik P. Kingma.
  "Weight Normalization: A Simple Reparameterization to Accelerate
   Training of Deep Neural Networks."
  https://arxiv.org/abs/1602.07868

  To enable the second form, set `normalize=True`.

  Args:
    processed_query: Tensor, shape `[batch_size, num_units]` to compare to keys.
    keys: Processed memory, shape `[batch_size, max_time, num_units]`.  可以理解为encoder的原始$h_i$
    normalize: Whether to normalize the score function.

  Returns:
    A `[batch_size, max_time]` tensor of unnormalized score values.
  """
  dtype = processed_query.dtype
  # Get the number of hidden units from the trailing dimension of keys
  num_units = keys.shape[2].value or array_ops.shape(keys)[2]
  # Reshape from [batch_size, ...] to [batch_size, 1, ...] for broadcasting.
  processed_query = array_ops.expand_dims(processed_query, 1)  # [batch_size, 1, num_units]
  v = variable_scope.get_variable(
      "attention_v", [num_units], dtype=dtype)
  if normalize:
    # Scalar used in weight normalization
    g = variable_scope.get_variable(
        "attention_g", dtype=dtype,
        initializer=math.sqrt((1. / num_units)))
    # Bias added prior to the nonlinearity
    b = variable_scope.get_variable(
        "attention_b", [num_units], dtype=dtype,
        initializer=init_ops.zeros_initializer())
    # normed_v = g * v / ||v||
    normed_v = g * v * math_ops.rsqrt(
        math_ops.reduce_sum(math_ops.square(v)))
    return math_ops.reduce_sum(
        normed_v * math_ops.tanh(keys + processed_query + b), [2])
  else:
    # keys + processed_query 这个操作是将processed_query加到对应batch的max_time个key上
    # 使用v做element wise product，还是[batch_size, max_time, num_units]，
    再对第二维度reduce_sum，得到[batch_size, mat_time]，即$e_{ij}$
    return math_ops.reduce_sum(v * math_ops.tanh(keys + processed_query), [2])
```
物理意义上，在encoder中查找与当前decoder状态最相关的部分，用于生成下一个词。其实也很抽象
无法解释这个相关是怎么计算的。
