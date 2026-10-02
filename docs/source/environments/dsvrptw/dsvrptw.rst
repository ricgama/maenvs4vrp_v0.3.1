:hide-toc:

==========
DSVRPTW
==========

Dynamic Stochastic Capacitated Vehicle Routing Problem with Time Windows (DSVRPTW)

The DSCVRPTW is a generalization of the classical Capacitated Vehicle Routing Problem with Time Windows, incorporating uncertain and dynamic changes during operation: not all information is known offline, as the environment evolves due to dynamic customer requests that arrive sequentially while vehicles are en route, and stochastic travel times that are random and unpredictable, requiring that all vehicle capacity and customer time window constraints be met despite these dynamic changes (see [Bon21]_).

Here's everything about DSVRPTW environment:

.. toctree::
    :maxdepth: 1
    
    agent-reward/agent-reward
    agent-selector/agent-selector
    environment/environment
    generation/generation
    observations/observations
