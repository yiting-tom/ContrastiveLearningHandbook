# methods package -- each SSL method gets its own sub-package.
# Importing sub-packages triggers register_method() calls in their __init__.py,
# making methods available to method_dispatcher() before it is called.

import methods.instance_discrimination  # noqa: F401
import methods.invariant_spread         # noqa: F401
import methods.simclr                   # noqa: F401
import methods.moco                     # noqa: F401
import methods.infomin                  # noqa: F401
import methods.swav                     # noqa: F401
import methods.barlow_twins             # noqa: F401
