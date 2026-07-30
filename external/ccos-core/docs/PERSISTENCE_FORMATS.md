# Persistence Formats

Persistence is local and written through atomic, synced replacement. Content-addressed cold records verify their digest before use; malformed or truncated records are rejected and surfaced to the audit log. Format compatibility is explicit per subsystem; operators must retain verified snapshots before migrations and test restore on a copy.
