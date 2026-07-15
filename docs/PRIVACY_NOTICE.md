# SciRust licensing — privacy notice (template)

> **Template — not legal advice.** This notice covers the personal data processed
> by SciRust's **licensing and provenance** mechanisms only. Fill in every
> `<placeholder>`, have it reviewed by your data-protection officer (DPO) and
> qualified counsel, and publish it at the URL referenced in
> [LICENSING.md](../LICENSING.md) § 5. It is drafted to meet the transparency
> requirements of Articles 13–14 of Regulation (EU) 2016/679 (**GDPR**). Provide a
> French-language version for French data subjects (Loi « Toubon »):
> [PRIVACY_NOTICE.fr.md](PRIVACY_NOTICE.fr.md).

## 1. Who is responsible (controller)

- **Controller:** `<legal name>`, `<address>`, `<company/registration no.>`.
- **Contact:** `<email>` · `<postal address>`.
- **Data-protection officer / privacy contact:** `<name / email, or "not appointed — contact above">`.
- **EU representative** (if the controller is established outside the EU, Art. 27):
  `<name / contact, or "not applicable">`.

## 2. Scope

This notice concerns **only** the licensing and provenance functions. SciRust's
computation does **not** process your workload data, inputs, or results: there is
no telemetry, no phone-home, and verification is performed entirely offline. It
does not apply to any separate website, sales, or support processing, which is
covered by `<link to your general privacy policy>`.

## 3. What personal data we process

| Data | Where it lives | Notes |
|---|---|---|
| Licensee identity (name / organisation) and licence id | inside the issued licence file | supplied by you when a licence is issued |
| One-time-signature **leaf serial** | licence file and signed artifacts | a per-issuance serial used for authenticity and leak attribution |
| **Pseudonymised machine fingerprint** (SHA-256), *only if you opt into node-locking* | inside the licence file | a salted hash of a machine identifier **you** supply; the raw identifier is never collected or stored |

No special-category data (Art. 9) is processed. No behavioural, usage, or
workload data is collected.

## 4. Purposes and legal bases (Art. 6)

| Purpose | Legal basis |
|---|---|
| Issue and verify licences; grant the entitlements you purchased | **Performance of the contract**, Art. 6(1)(b) |
| Authenticity of provenance marks and **attribution of leaked / infringing copies** (anti-piracy) | **Legitimate interests**, Art. 6(1)(f) — protecting the controller's intellectual property; balanced against your interests, and limited to a serial and a salted hash (see § 3) |
| Comply with legal obligations (e.g. accounting, responding to lawful requests) | **Legal obligation**, Art. 6(1)(c) |

Where the basis is legitimate interests, you may **object** (Art. 21); see § 7.

## 5. Recipients and international transfers

- **Recipients:** `<none / your processors, e.g. hosting or CRM — list them>`. Any
  processor acts under a written Art. 28 agreement.
- **International transfers:** `<none / if any, the safeguard used — adequacy
  decision or Art. 46 tool such as SCCs>`.
- Verification runs offline on your systems; the controller does not receive data
  from your running deployment.

## 6. Retention

- Licence records (identity, licence id, leaf serial): `<e.g. duration of the
  licence + <n> years>` for contract, warranty, and IP-enforcement purposes.
- Data no longer needed is deleted or anonymised. `<Set concrete periods.>`

## 7. Your rights

Subject to the conditions in the GDPR, you may request **access** (Art. 15),
**rectification** (Art. 16), **erasure** (Art. 17), **restriction** (Art. 18),
**portability** (Art. 20), and **object** to processing based on legitimate
interests (Art. 21). To exercise them, contact `<privacy contact from § 1>`. You
also have the right to lodge a complaint with a supervisory authority — in France,
the **CNIL** (www.cnil.fr) — or the authority of your habitual residence.

## 8. Is providing this data mandatory?

Providing the licensee identity is a **contractual requirement** to issue and
verify a licence; without it a licence cannot be granted. Node-locking is
**optional** — if you do not opt in, no machine fingerprint is processed.

## 9. Automated decision-making

The licence check is a deterministic validity verification that grants or refuses
a software capability and is **fully recoverable** by installing a valid licence.
The controller does not carry out automated decision-making producing legal or
similarly significant effects on you within the meaning of **Art. 22**.

## 10. Source of the data (Art. 14)

Where personal data (e.g. the identity of your designated contact) reaches the
controller from the licensee rather than directly from the data subject, its
source is the **licensee organisation** that requested the licence.

## 11. Changes

`<Version / date>`. We will update this notice as needed and indicate the date of
the latest revision here.
