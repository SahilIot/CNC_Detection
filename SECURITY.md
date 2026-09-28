# Security Policy

## Supported versions

Security fixes are provided for the latest version on the repository's
default branch. Older revisions may not receive security updates.

## Reporting a vulnerability

Please do not report suspected vulnerabilities in a public issue. Use GitHub's
**Report a vulnerability** feature in the repository's Security tab (private
vulnerability reporting) when it is available. If it is not enabled, contact
the repository maintainers privately through GitHub before sharing technical
details.

Include, when possible:

- Affected commit, version, or component.
- A concise description of the impact and conditions required.
- Reproduction steps or a proof of concept that does not access real cameras,
  accounts, or data without authorization.
- Any proposed mitigation.

Do not attach RTSP credentials, `.env` files, camera footage, real user or
site data, database files, or other secrets. Redact sensitive information
from logs and examples.

Maintainers will acknowledge reports as soon as practical, investigate the
issue, and coordinate any fix and disclosure with the reporter. Please allow
reasonable time for a fix before public disclosure.

## Security considerations

This project is intended for trusted, on-premises networks. The dashboard and
backend do not implement user authentication or role-based access control.
Restrict network access to both services, protect the SQLite database (which
can contain camera credentials), and do not expose the services directly to
the public internet.
