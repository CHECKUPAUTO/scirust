## 2026-05-29 - [Harden monitor API against DoS and Info Leakage]
**Vulnerability:** Information leakage in API error responses and unbounded input 'limit' parameter leading to potential DoS.
**Learning:** Returning 'e.to_string()' in HTTP 500 responses exposes internal system details. Unvalidated integer parameters used for database limits allow for resource exhaustion.
**Prevention:** Enforce upper bounds on all user-provided 'limit' parameters. Use generic error messages in public API responses while logging actual errors internally for debugging.
