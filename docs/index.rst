cartbctx documentation
======================

**cartbctx** implements *Classification and Regression Trees with Backward
Compatible Tree-based Explanations under Retraining Scenarios* (CART-BCTX),
a natural extension of the CART algorithm that supports backward compatible
tree-based explanations.

The main API conforms with scikit-learn: estimators expose ``fit`` /
``predict`` / ``score``, plus an ``update`` method that retrains the tree on a
new dataset while suppressing explanation changes for backward compatibility
with the previously fitted tree.

.. code-block:: python

   import numpy as np
   from sklearn.datasets import load_iris
   from sklearn.model_selection import train_test_split
   from cartbctx import DecisionTreeClassifierBCTX

   X, y = load_iris(return_X_y=True, as_frame=True)
   X_old, X_tmp, y_old, y_tmp = train_test_split(X, y, train_size=0.33, random_state=0)
   X_add, X_tst, y_add, y_tst = train_test_split(X_tmp, y_tmp, train_size=0.5, random_state=0)
   X_new = np.concatenate([X_old, X_add], axis=0)
   y_new = np.concatenate([y_old, y_add], axis=0)

   # 1. Train the initial (old) tree.
   clf = DecisionTreeClassifierBCTX(bctx_mode='MSE', bctx_lambda=0.005)
   clf.fit(X_old, y_old)

   # 2. Retrain on the new dataset with BCTX regularization for backward
   #    compatibility with the old tree.
   clf.update(X_new, y_new)

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
