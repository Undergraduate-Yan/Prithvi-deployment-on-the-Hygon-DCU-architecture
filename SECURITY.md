# Security policy

Do not commit SSH passwords, private hostnames, cluster IP addresses, API tokens, `.env` files, private keys, or signed download URLs.

If a credential is found in Git history:

1. revoke or rotate it immediately;
2. remove it from the current tree;
3. rewrite Git history with an appropriate tool;
4. notify all collaborators who cloned the affected history.

Security reports should be sent privately to the repository owner rather than opened as a public issue. Replace this paragraph with a real private contact before public release.

