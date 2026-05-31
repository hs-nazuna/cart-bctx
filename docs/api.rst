API Reference
=============

This page documents the public API of :mod:`cartbctx`.

.. currentmodule:: cartbctx

.. autosummary::
   :nosignatures:

   DecisionTreeClassifierBCTX
   DecisionTreeRegressorBCTX

Classifier
----------

.. autoclass:: cartbctx.DecisionTreeClassifierBCTX
   :members:
   :inherited-members:
   :show-inheritance:
   :exclude-members: cost_complexity_pruning_path, prune_tree, get_metadata_routing, set_fit_request, set_score_request, set_params, get_params

Regressor
---------

.. autoclass:: cartbctx.DecisionTreeRegressorBCTX
   :members:
   :inherited-members:
   :show-inheritance:
   :exclude-members: cost_complexity_pruning_path, prune_tree, get_metadata_routing, set_fit_request, set_score_request, set_params, get_params
