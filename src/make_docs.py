"""Writes docs/feature_dictionary.md from features.FEATURE_DICT (so code and documentation never differ)."""
from config import ROOT
from features import FEATURE_DICT, CATEGORICAL

lines = ["# Feature dictionary", "",
         "All features are computed per order and are available when the order is approved "
         "(see the leakage rules in README).", "",
         "| Feature | Type | Source tables | How it is computed | Why it may matter |",
         "|---|---|---|---|---|"]
for name, (src, how, why) in FEATURE_DICT.items():
    kind = "categorical" if name in CATEGORICAL else "numeric"
    lines.append(f"| `{name}` | {kind} | {src} | {how} | {why} |")
lines += ["", "Excluded on purpose (outcome information): `order_status`, `order_delivered_customer_date`, "
          "`order_delivered_carrier_date`, `actual_delivery_days`, `delay_days`. "
          "`shipping_limit_date` is also left out because we cannot prove when it was set."]
(ROOT / "docs" / "feature_dictionary.md").write_text("\n".join(lines) + "\n")
print("docs/feature_dictionary.md written")
