# Schrodinger_Feynman_Kac
This code simulates the Schrödinger equation using the Feynman-Kac formula derived in https://arxiv.org/abs/2409.16519. The original implementation, built with TensorFlow 2.4.1, supported only the one-dimensional setting. In contrast, this version is implemented in PyTorch and supports both one-dimensional and high-dimensional cases.
By default, the linear case is used. To enable the nonlinear case, uncomment the relevant import statement from nonlinear_case_related_functions.
