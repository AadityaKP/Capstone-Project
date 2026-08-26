"""The founder test suites (test_whatif, test_founder_*, test_api) were
written against the founder configuration and assert its behaviour - real
monthly_burn, scale-aware curves, scaled floors. SIM_PROFILE now defaults to
review2, so pin the profile they test unless the environment already chose
one. Review2-profile behaviour is covered by the parity harnesses in the
review2-sim-frontend verification, not by these suites.
"""

import os

os.environ.setdefault("SIM_PROFILE", "founder")
