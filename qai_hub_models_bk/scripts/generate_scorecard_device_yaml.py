# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import sys

from qai_hub_models.configs.devices_and_chipsets_yaml import DevicesAndChipsetsYaml
from qai_hub_models.utils.envvars import IsOnCIEnvvar

if __name__ == "__main__":
    """
    Re-generate and save a DevicesAndChipsetsYaml configuration from the current
    set of devices / runtimes that are valid in AI Hub Models perf.yaml files.
    """
    if not IsOnCIEnvvar.get():
        print(
            "This script must be run on PROD WORKBENCH with a PUBLIC USER (no access to un-published devices). We recommend using qaihm_bot@qti.qualcomm.com as the user."
        )
        if input("Are you set up accordingly? (y/n) --> ").lower() not in ["y", "yes"]:
            print("exiting")
            sys.exit(0)

    DevicesAndChipsetsYaml.from_all_runtimes_and_devices().save()
