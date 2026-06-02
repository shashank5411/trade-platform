#!/usr/bin/env python3
import os

import aws_cdk as cdk

from trade_platform.trade_platform_stack import TradePlatformStack


app = cdk.App()

TradePlatformStack(app, "TradePlatformStack-dev", env_name="dev")
TradePlatformStack(app, "TradePlatformStack-prod", env_name="prod")

app.synth()
