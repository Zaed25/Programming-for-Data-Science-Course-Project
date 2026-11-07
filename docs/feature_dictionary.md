# Feature dictionary

All features are computed per order and are available when the order is approved (see the leakage rules in README).

| Feature | Type | Source tables | How it is computed | Why it may matter |
|---|---|---|---|---|
| `n_items` | numeric | items | number of item rows in the order | more items, more handling |
| `n_products` | numeric | items | distinct products | mixed orders |
| `n_sellers` | numeric | items | distinct sellers | multi-seller orders ship in several packages |
| `multi_seller` | numeric | items | 1 if n_sellers > 1 | explicit flag for multi-seller orders |
| `n_categories` | numeric | items+products | distinct categories | order complexity |
| `total_price` | numeric | items | sum of item prices | order value |
| `total_freight` | numeric | items | sum of freight values | freight reflects distance and size |
| `freight_ratio` | numeric | items | freight / (price + freight) | expensive shipping relative to value = far/heavy |
| `max_item_price` | numeric | items | most expensive item | valuable items may be handled differently |
| `total_weight_g` | numeric | items+products | sum of product weights | heavy orders |
| `max_weight_g` | numeric | items+products | heaviest product | one bulky item can slow the order |
| `total_volume_cm3` | numeric | items+products | sum of length*height*width | bulky orders |
| `planned_days` | numeric | orders | estimated date - approval date (days) | the promise given to the customer |
| `approval_delay_h` | numeric | orders | approval - purchase (hours) | slow payment approval (e.g. boleto) |
| `approve_weekday` | numeric | orders | day of week of approval | weekend approvals wait longer |
| `approve_hour` | numeric | orders | hour of approval | late approvals ship next day |
| `approve_month` | numeric | orders | month of approval | seasonality (Black Friday, Christmas) |
| `all_sellers_same_state` | numeric | items+sellers+customers | 1 if every seller is in the customer state | short routes; handles multi-seller orders |
| `main_seller_same_state` | numeric | items+sellers+customers | main seller state == customer state | short route |
| `n_seller_states` | numeric | items+sellers | distinct seller states | packages come from several states |
| `payment_total` | numeric | payments | sum of payment values | order value incl. interest |
| `n_payment_records` | numeric | payments | number of payment rows | split payments |
| `max_installments` | numeric | payments | max installments | payment behaviour |
| `uses_voucher` | numeric | payments | 1 if any voucher | payment behaviour |
| `seller_prior_orders` | numeric | history | main seller's orders delivered before this approval | seller experience |
| `seller_prior_late_rate` | numeric | history | late share of those past orders (NaN if none) | seller reliability |
| `customer_prior_orders` | numeric | history | orders by the same person before this purchase | repeat customer |
| `customer_state` | categorical | customers | state of the customer | remote states (north/north-east) |
| `main_seller_state` | categorical | items+sellers | state of the main seller | origin of the package |
| `main_category` | categorical | items+products | category of the most expensive item | product type |
| `main_payment_type` | categorical | payments | payment type with the largest value | boleto vs card |

Excluded on purpose (outcome information): `order_status`, `order_delivered_customer_date`, `order_delivered_carrier_date`, `actual_delivery_days`, `delay_days`. `shipping_limit_date` is also left out because we cannot prove when it was set.
