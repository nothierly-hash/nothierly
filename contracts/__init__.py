"""Contrats d'interface entre le moteur et le monde extérieur.

``trading_signal`` : message Python/JSON moteur -> workers de plateformes. ``NothierlyVault.sol`` :
contrat Solidity, sans lien avec le précédent. Ce package ne doit rien importer de ``core``/``src``.
"""
